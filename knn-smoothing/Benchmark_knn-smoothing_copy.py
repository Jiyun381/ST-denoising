import anndata as ad
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import time
import sys
from math import log, ceil
import pandas as pd

# knn_smoothing 함수 정의
def _median_normalize(X):
    num_transcripts = np.sum(X, axis=0)
    median_umi = np.median(num_transcripts)
    if median_umi == 0:
        print("Warning: median UMI is 0, skip normalization.")
        return X

    zero_mask = (num_transcripts == 0)
    scale_factors = np.ones_like(num_transcripts, dtype=float)
    nonzero_mask = ~zero_mask
    scale_factors[nonzero_mask] = median_umi / num_transcripts[nonzero_mask]
    X_norm = X * scale_factors
    return X_norm
 
def _freeman_tukey_transform(X):
    return np.sqrt(X) + np.sqrt(X + 1)

def _calculate_pc_scores(matrix, d, seed=None):
    # 데이터 유효성 체크
    if matrix.size == 0:
        print("Warning: Empty matrix, skipping PCA.")
        return matrix

    tmatrix = _median_normalize(matrix)
    tmatrix = _freeman_tukey_transform(tmatrix)

    p, n = tmatrix.shape
    num_pcs = min(p, n - 1)
    # d가 num_pcs보다 클 경우 줄이기
    if d > num_pcs:
        print(f"Warning: d={d} > num_pcs={num_pcs}, reducing d to {num_pcs}.")
        d = num_pcs
    if d == 0:
        print("Warning: No PCs available (d=0), returning original matrix.")
        return tmatrix

    pca = PCA(n_components=d, svd_solver='randomized', random_state=seed)
    t0 = time.time()
    tmatrix = pca.fit_transform(tmatrix.T).T
    t1 = time.time()
    var_explained = np.cumsum(pca.explained_variance_ratio_)[-1] if pca.explained_variance_ratio_.size > 0 else 0
    print('\tPCA took %.1f s.' % (t1 - t0))
    print('\tThe fraction of variance explained by the top %d PCs is %.1f %%.' % (d, 100 * var_explained))
    return tmatrix

def _calculate_pairwise_distances(X, num_jobs=1):
    # 데이터 유효성 체크
    if X.size == 0:
        print("Warning: Empty data for pairwise distances, returning empty array.")
        return np.array([])
    return pairwise_distances(X.T, n_jobs=num_jobs, metric='euclidean')

def knn_smoothing(X, k, d=10, dither=0.03, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if not (X.dtype == np.float64 or X.dtype == np.float32):
        raise ValueError('X must contain floating point values! Try X = np.float64(X).')

    p, n = X.shape
    if n == 0 or p == 0:
        print("Warning: Empty matrix (no cells or no genes). Returning original X.")
        return X

    num_pcs = min(p, n - 1)
    if k < 1 or k > n:
        raise ValueError(f'k must be between 1 and {n}.')
    if d < 1:
        print("Warning: d < 1, no PCA dimension to reduce, returning original X.")
        return X
    if d > num_pcs:
        print(f"Warning: d={d} > num_pcs={num_pcs}, reducing d to {num_pcs}.")
        d = num_pcs
    if d == 0:
        print("Warning: No PCs available after adjustment, returning original X.")
        return X

    print('Performing kNN-smoothing with k=%d, d=%d, and dither=%.3f...' % (k, d, dither))
    sys.stdout.flush()

    t0_total = time.time()

    if k == 1:
        num_steps = 0
    else:
        num_steps = ceil(log(k) / log(2))

    S = X.copy()

    for t in range(1, num_steps + 1):
        k_step = min(pow(2, t), k)
        print('Step %d/%d: Smooth using k=%d' % (t, num_steps, k_step))
        sys.stdout.flush()

        Y = _calculate_pc_scores(S, d, seed=seed)
        if Y.size == 0:
            print("Warning: No data for PCA, skipping smoothing step.")
            continue

        if dither > 0:
            for l in range(d):
                ptp = np.ptp(Y[l, :])
                dy = (np.random.rand(Y.shape[1]) - 0.5) * ptp * dither
                Y[l, :] = Y[l, :] + dy

        t0 = time.time()
        D = _calculate_pairwise_distances(Y)
        t1 = time.time()
        print('\tCalculating pair-wise distance matrix took %.1f s.' % (t1 - t0))
        sys.stdout.flush()

        if D.size == 0:
            print("Warning: Empty distance matrix, no smoothing this step.")
            continue

        t0 = time.time()
        A = np.argsort(D, axis=1, kind='mergesort')
        for j in range(X.shape[1]):
            ind = A[j, :k_step]
            S[:, j] = np.sum(X[:, ind], axis=1)

        t1 = time.time()
        print('\tCalculating the smoothed expression matrix took %.1f s.' % (t1 - t0))
        sys.stdout.flush()

    t1_total = time.time()
    print('kNN-smoothing finished in %.1f s.' % (t1_total - t0_total))
    sys.stdout.flush()

    return S

def main():
    bench_data_dir = '/home/s2022310812/bench_data'        # bench_data 경로 [수정]
    dropout_data_dir = '/home/s2022310812/dropout_data'    # dropout_data 경로 [수정]
    datasets = [                                           # [수정]
        "IDC",
        # "COAD",
        # "READ",
        # "LYMPH_IDC",
    ]
    model = "knn-smoothing"                                # [수정] 모델명
    percents = [                                           # [수정] dropout percentage
        '30%',
        # '20%',
        # '30%'
    ]
    samples = [                                            # [수정] dropout percentage data version
          '1'
        # '2',
        # '3',
        # '4',
         
    ]
    
    trials = range(2, 6)  # 1부터 5까지의 트라이얼 번호

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # [수정] GPU 디바이스 설정
    print(f"Using device: {device}")

    for trial in trials:
        print(f"\n===== Starting Trial {trial} =====")
        seed = trial  # 트라이얼 번호를 시드로 사용

        for dataset in datasets:
            for percent in percents:
                for sample in samples:
                    man_sc = {}  # 각 샘플마다 초기화
                    bench_path = os.path.join(bench_data_dir, dataset, 'adata')
                    dropout_path = os.path.join(dropout_data_dir, dataset, f"{percent}-{sample}")

                    # 'bench_path'와 'dropout_path'가 존재하는지 확인
                    if not os.path.exists(bench_path):
                        print(f"Error: Benchmark path '{bench_path}' does not exist. Skipping.")
                        continue
                    if not os.path.exists(dropout_path):
                        print(f"Error: Dropout path '{dropout_path}' does not exist. Skipping.")
                        continue

                    adata_names = [f for f in os.listdir(bench_path) if os.path.isfile(os.path.join(bench_path, f))]

                    for adata_name in adata_names:
                        # 데이터 로딩 및 전처리
                        adata = ad.read_h5ad(os.path.join(bench_path, adata_name))
                        sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=False)
                        sc.pp.log1p(adata)

                        adata_drop = ad.read_h5ad(os.path.join(dropout_path, adata_name))
                        sc.pp.normalize_total(adata_drop, target_sum=1e6, exclude_highly_expressed=False)
                        sc.pp.log1p(adata_drop)

                        # 'dropped' 레이어가 있는지 확인
                        if 'dropped' not in adata_drop.layers:
                            print(f"Warning: 'dropped' layer not found in {adata_name}. Skipping.")
                            man_sc[dataset + "/" + adata_name] = np.nan
                            continue

                        dropped_mask = torch.tensor(adata_drop.layers['dropped'], dtype=torch.bool, device=device)

                        """
                        [수정]
                        여기서부터 [여기까지] adata_drop.X에 knn_smoothing 모델을 적용하여 adata2에 저장합니다.
                        [주의] adata2는 adata.X에 해당하는 파일을 저장해야 합니다.
                        """

                        # knn_smoothing 모델 적용
                        if isinstance(adata_drop.X, csr_matrix):
                            X_drop = adata_drop.X.toarray()
                        else:
                            X_drop = adata_drop.X

                        X_drop = X_drop.astype(np.float64)  # knn_smoothing에 맞게 float64
                        X_drop_t = X_drop.T  # (genes, cells)

                        k = 10
                        d = 10
                        dither = 0.03
                        # seed = 0  # 제거: 시드가 고정되지 않도록

                        # 데이터 유효성 체크
                        if X_drop_t.size == 0:
                            print("Warning: No data in adata_drop, skipping smoothing and ARI.")
                            man_sc[dataset + "/" + adata_name] = np.nan
                            continue

                        S = knn_smoothing(X_drop_t, k=k, d=d, dither=dither, seed=seed)  # 시드 전달

                        adata2 = S.T  # (cells, genes)

                        # adata2의 스케일을 adata에 맞추기 위해 정규화 및 로그 변환 적용
                        adata2_sum = adata2.sum(axis=1, keepdims=True)
                        zero_mask = adata2_sum == 0
                        adata2_sum[zero_mask] = 1  # 0으로 나누는 것을 방지
                        adata2 = adata2 / adata2_sum * 1e6
                        adata2 = np.log1p(adata2)

                        """
                        [여기까지]
                        """

                        # Step 1: Convert adata.X to float32 before creating the PyTorch tensor
                        if isinstance(adata.X, np.ndarray):
                            adata_X = adata.X.astype(np.float32) if adata.X.dtype == np.uint16 else adata.X.astype(np.float32)
                            adata_tensor = torch.tensor(adata_X, dtype=torch.float32, device=device)
                        elif isinstance(adata.X, csr_matrix):
                            adata_X = adata.X.toarray().astype(np.float32)
                            adata_tensor = torch.tensor(adata_X, dtype=torch.float32, device=device)
                        else:
                            print('Error: Unsupported adata.X 데이터 형식.')
                            man_sc[dataset + "/" + adata_name] = np.nan
                            continue  # Unsupported format

                        # Step 2: Convert adata2 to float32 before creating the PyTorch tensor
                        if isinstance(adata2, np.ndarray):
                            adata2 = adata2.astype(np.float32) if adata2.dtype == np.uint16 else adata2.astype(np.float32)
                            adata2_tensor = torch.tensor(adata2, dtype=torch.float32, device=device)
                        elif isinstance(adata2, csr_matrix):
                            adata2 = adata2.toarray().astype(np.float32)
                            adata2_tensor = torch.tensor(adata2, dtype=torch.float32, device=device)
                        else:
                            print('Error: Unsupported adata2 데이터 형식.')
                            man_sc[dataset + "/" + adata_name] = np.nan
                            continue  # Unsupported format

                        # Step 3: Get the indices of the dropped values (where the mask is True)
                        dropped_indices = dropped_mask.nonzero(as_tuple=True)

                        # Step 4: Extract the corresponding values for dropped spots
                        adata_values = adata_tensor[dropped_indices]
                        adata2_values = adata2_tensor[dropped_indices]

                        # Step 5: Convert to CPU and calculate Manhattan distance
                        adata_values_cpu = adata_values.cpu().numpy()
                        adata2_values_cpu = adata2_values.cpu().numpy()

                        manhattan_distance = np.abs(adata_values_cpu - adata2_values_cpu).sum()
                        num_values = adata_values_cpu.size
                        manhattan_score = manhattan_distance / num_values if num_values > 0 else np.nan

                        # Store the result in the dictionary
                        man_sc[dataset + "/" + adata_name] = manhattan_score

                        print(man_sc)

                    # 결과 저장
                    row_name = f"Trial : {trial}"
                    my_dir = os.path.join('/home/s2022310812/bench_results', model, dataset)
                    csv_path = os.path.join(my_dir, f"{percent}-{sample}_result.csv")
                    # Ensure the directory exists
                    os.makedirs(my_dir, exist_ok=True)

                    # Extract column names and values
                    columns = [key.split("/")[-1].split(".")[0] for key in man_sc.keys()]
                    values = list(man_sc.values())

                    # Create a DataFrame for the new row
                    data = pd.DataFrame([values], columns=columns, index=[row_name])

                    # Check if the file exists
                    if os.path.exists(csv_path):
                        # Read the existing CSV
                        try:
                            existing_data = pd.read_csv(csv_path, index_col=0)
                            # Append the new row
                            updated_data = pd.concat([existing_data, data])
                        except Exception as e:
                            print(f"Error reading existing CSV '{csv_path}': {e}")
                            updated_data = data
                    else:
                        # If the file doesn't exist, use the new DataFrame
                        updated_data = data

                    # Save back to CSV
                    try:
                        updated_data.to_csv(csv_path)
                        print(f"Saved results to '{csv_path}'")
                    except Exception as e:
                        print(f"Error saving CSV '{csv_path}': {e}")

        print(f"===== Completed Trial {trial} =====\n")

    print("Benchmarking completed.")

if __name__ == "__main__":
    main()
    
