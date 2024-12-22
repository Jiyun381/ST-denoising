import anndata as ad
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch
from scipy.sparse import csr_matrix
import scanpy as sc

import ot
import pandas as pd
import math
import alphashape
from shapely.geometry import Point, Polygon
from multiprocessing import Pool, cpu_count
import sys
import csv
from scipy import sparse
import heapq


class SpotGF:
    def __init__(self, gem_path, binsize, proportion, auto_threshold, lower, upper, max_iterations, outpath, visualize, spot_size, alpha):
        self.gem_path = gem_path
        self.binsize = binsize
        self.proportion = proportion
        self.auto_threshold = auto_threshold
        self.lower = lower
        self.upper = upper
        self.max_iterations = max_iterations
        self.outpath = outpath
        self.visualize = visualize
        self.spot_size = spot_size
        self.alpha = alpha
        os.makedirs(outpath, exist_ok=True)  # Ensure output directory exists
        # Removed os.chdir(outpath) to avoid changing global working directory
        print("Input file:", gem_path)
        print("Output path:", outpath)
        print("Denoising resolution:", binsize)

    def gem2adata(self, result):
        result['x_y'] = result['x'].astype('str') + '_' + result['y'].astype('str')
        result = result[['x_y', 'geneID', 'MIDCount']]
        cell_list = result["x_y"].astype('category')
        gene_list = result["geneID"].astype('category')
        data = result["MIDCount"].to_numpy()
        row = cell_list.cat.codes.to_numpy()
        col = gene_list.cat.codes.to_numpy()
        obs = pd.DataFrame(index=cell_list.cat.categories)
        var = pd.DataFrame(index=gene_list.cat.categories)
        X = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var)))
        adata = ad.AnnData(X, obs=obs, var=var)
        coords = np.array([list(map(int, idx.split('_'))) for idx in adata.obs.index])
        adata.obsm['spatial'] = coords
        return adata

    def convert_x_y_to_numeric(self, df):
        # Convert 'x' and 'y' to numeric, coercing errors to NaN
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

        # Identify non-integer values
        non_integer_x = df['x'].dropna() % 1 != 0
        non_integer_y = df['y'].dropna() % 1 != 0

        if non_integer_x.any() or non_integer_y.any():
            print("Warning: Some 'x' or 'y' values have decimal parts. They will be rounded to the nearest integer.")
            # Round the values to the nearest integer
            df['x'] = df['x'].round().astype('Int64')
            df['y'] = df['y'].round().astype('Int64')
        else:
            # If all values are integers, safely cast to 'Int64'
            df['x'] = df['x'].astype('Int64')
            df['y'] = df['y'].astype('Int64')

        # Drop rows with NaNs in 'x' or 'y'
        df = df.dropna(subset=['x', 'y'])

        # Finally, cast to standard 'int64' (non-nullable)
        df = df.astype({'x': 'int64', 'y': 'int64'})

        return df

    def open_gem(self, gem_path):
        with open(gem_path, 'r') as file:
            sample_data = file.read(1024)
            dialect = csv.Sniffer().sniff(sample_data)
        if gem_path.endswith('.gz'):
            ex_raw = pd.read_csv(gem_path, delimiter=dialect.delimiter, compression='gzip', comment='#')
        else:
            ex_raw = pd.read_csv(gem_path, delimiter=dialect.delimiter, comment='#')
        # Standardize MIDCount column name
        ex_raw = ex_raw.rename(columns={
            'UMICount': 'MIDCount',
            'UMICounts': 'MIDCount',
            'MIDCounts': 'MIDCount'
        })
        ex_raw = self.convert_x_y_to_numeric(ex_raw)
        return ex_raw

    def preparedata(self, ex_raw):
        count = ex_raw['geneID'].value_counts()
        gsave = count[count > 10].index.tolist()
        ex_raw = ex_raw[ex_raw.geneID.isin(gsave)].reset_index(drop=True)
        all_gene = ex_raw['geneID'].unique()
        print("Valid genes Number:", len(all_gene))
        ex2 = ex_raw.groupby(['geneID', 'x', 'y'], as_index=False)['MIDCount'].sum()
        all_cell = ex_raw[['x', 'y']].drop_duplicates().values
        print("Valid cells number:", len(all_cell))
        return ex2, all_cell, all_gene

    def grid_downsample(self, points, num_points, mask_area, alpha_shape):
        if num_points == 0 or len(points) == 0:
            return points
        if mask_area == 0:
            return points

        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y

        bin_size = np.sqrt(mask_area / num_points)
        if bin_size == 0 or np.isinf(bin_size):
            return points

        num_cols = math.ceil(width / bin_size)
        num_rows = math.ceil(height / bin_size)
        if num_cols <= 0 or num_rows <= 0:
            return points

        grid = np.zeros((num_rows, num_cols), dtype=int)
        cols = ((points[:, 0] - min_x) / bin_size).astype(int)
        rows = ((points[:, 1] - min_y) / bin_size).astype(int)

        for col, row in zip(cols, rows):
            if 0 <= col < num_cols and 0 <= row < num_rows:
                grid[row, col] += 1

        output_points = []
        for row in range(num_rows):
            for col in range(num_cols):
                if grid[row, col] > 0:
                    x = (col + 0.1) * bin_size + min_x
                    y = (row + 0.1) * bin_size + min_y
                    if alpha_shape.contains(Point(x, y)):
                        output_points.append([x, y])
        output_points = np.array(output_points)
        return output_points

    def _median_normalize(self, X):
        num_transcripts = np.sum(X, axis=0)
        median_umi = np.median(num_transcripts)
        if median_umi == 0:
            return X

        zero_mask = (num_transcripts == 0)
        scale_factors = np.ones_like(num_transcripts, dtype=float)
        nonzero_mask = ~zero_mask
        scale_factors[nonzero_mask] = median_umi / num_transcripts[nonzero_mask]
        X_norm = X * scale_factors
        return X_norm

    def _process_gene(self, args):
        n, grouped_ex2, all_cell, alpha_shape, mask_area = args
        try:
            gene_c = grouped_ex2.get_group(n)
            # Compute weighted counts
            wnc = gene_c.groupby(['x', 'y']).size().reset_index(name='count')
            source_point = wnc[['x', 'y']].to_numpy()
            source_w = wnc['count'].to_numpy(dtype='float64')

            if len(source_point) > 5000:
                source_point = self.grid_downsample(source_point, 5000, mask_area, alpha_shape)
                if len(source_point) == 0:
                    return (n, 0.0)
                source_w = np.ones(len(source_point))

            if len(source_point) == 0:
                return (n, 0.0)

            target_point = self.grid_downsample(all_cell, len(source_point), mask_area, alpha_shape)
            if len(target_point) == 0:
                target_point = source_point.copy()

            if len(target_point) == 0:
                return (n, 0.0)

            target_w = np.ones(len(target_point), dtype='float64') * (sum(source_w) / len(target_point))
            M = ot.dist(source_point, target_point, metric='euclidean')
            result2 = ot.emd2(source_w, target_w, M, numItermax=1000000) / (len(source_point) if len(source_point) > 0 else 1)
            return (n, result2)
        except Exception as e:
            print(f"Error processing gene {n}: {e}")
            return (n, 0.0)

    def calculate_ot(self, all_gene, ex2, all_cell, alpha_shape):
        emd = []
        gene = []
        grouped_ex2 = ex2.groupby('geneID')
        mask_area = alpha_shape.area

        # Prepare arguments for parallel processing
        args_list = [(n, grouped_ex2, all_cell, alpha_shape, mask_area) for n in all_gene]

        # Determine the number of processes to use
        num_processes = max(cpu_count() - 1, 1)  # Leave one CPU free

        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_gene, args_list)

        # Collect results
        for gene_id, score in results:
            gene.append(gene_id)
            emd.append(score)

        result = np.array([gene, emd])
        return result

    def calculate_GFscore(self, gem_path, binsize, alpha, max_iterations, lower, upper):
        ex_raw = self.open_gem(gem_path)
        if binsize == 1:
            print('Process with resolution cell bin data')
            ex_raw.rename(columns={'cen_y': 'y'}, inplace=True)  # For cell-bin data
            ex_raw.rename(columns={'cen_x': 'x'}, inplace=True)
        else:
            print('Process with resolution binsize', str(binsize), 'data')
            ex_raw['x'] = ex_raw['x'].map(lambda x: int(x / binsize))
            ex_raw['y'] = ex_raw['y'].map(lambda y: int(y / binsize))

        ex2, all_cell, all_gene = self.preparedata(ex_raw)

        if len(all_cell) < 4:
            if len(all_cell) == 0:
                alpha_shape = Polygon()
            elif len(all_cell) == 1:
                alpha_shape = Point(all_cell[0][0], all_cell[0][1]).buffer(1)
            else:
                poly_points = [tuple(p) for p in all_cell]
                alpha_shape = Polygon(poly_points).buffer(1)
        else:
            if alpha == 0:
                try:
                    alpha_use = alphashape.optimizealpha(all_cell, max_iterations, lower, upper)
                except:
                    alpha_use = 1.0
            else:
                alpha_use = alpha
            try:
                alpha_shape = alphashape.alphashape(all_cell, alpha_use)
            except:
                min_x, min_y = np.min(all_cell, axis=0)
                max_x, max_y = np.max(all_cell, axis=0)
                alpha_shape = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

        # Apply multiprocessing in calculate_ot
        result = self.calculate_ot(all_gene, ex2, all_cell, alpha_shape)
        GF_df = pd.DataFrame(result.T, columns=['geneID', 'SpotGF_score'])
        GF_df.to_csv(os.path.join(self.outpath, 'SpotGF_scores.txt'), sep='\t', index=False)
        print("Finished saving SpotGF_scores.txt")
        return GF_df

    def cal_threshold(self, emd2):
        x = np.arange(len(emd2))
        emd2 = np.array(emd2, dtype=float)
        dydx = np.gradient(emd2, x)
        mean_dydx = np.mean(dydx)
        dydx[dydx > mean_dydx] = mean_dydx
        cur = np.polyfit(x, dydx, 10)
        p1 = np.poly1d(cur)
        dy = np.gradient(p1(x), x)
        max_idx = np.argmax(dy)
        max_point = (x[max_idx], emd2[max_idx])
        for i in [15, 20, 25, 30]:
            if x[max_idx] <= np.median(x):
                cur = np.polyfit(x, dydx, i)
                p1 = np.poly1d(cur)
                dy = np.gradient(p1(x), x)
                max_idx = np.argmax(dy)
                max_point = (x[max_idx], emd2[max_idx])
        print('Threshold:', max_point)
        return max_point

    def expression_figure(self, adata, save_path, spot_size):
        if adata.n_vars > 200:
            adata.var["mt"] = adata.var_names.str.startswith("MT")
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=(50, 100, 200), inplace=True, log1p=True)
            fig = sc.pl.spatial(adata, color='total_counts', spot_size=spot_size, title='Total_counts',
                                show=False, return_fig=True, cmap='Spectral_r')
            ax = fig.axes[0]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(width=3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)
            fig.savefig(save_path, dpi=300)
            return adata
        else:
            print("Warning: Gene numbers below 200 cannot visualize denoised data")

    def generate_GFgem(self, gem_path, GF_df, proportion, auto_threshold, visualize, spot_size, all_genes):
        ex_raw = self.open_gem(gem_path)
        count = ex_raw['geneID'].value_counts()
        gsave_high = count[count <= 10].index.tolist()
        GF_df['SpotGF_score'] = GF_df['SpotGF_score'].astype(float)
        GF_df = GF_df.sort_values(by='SpotGF_score')
        emd2 = GF_df['SpotGF_score'].tolist()

        # Auto threshold
        if auto_threshold:
            print("Generate SpotGF-denoised data based on automatic threshold")
            thred = self.cal_threshold(emd2)
            save_gene = GF_df[GF_df['SpotGF_score'] >= float(thred[1])]['geneID'].tolist()
            save = set(save_gene + gsave_high)
            # Retain all genes; filtered genes have MIDCount=0
            ex_raw['Selected'] = ex_raw['geneID'].isin(save)
            ex_raw.loc[~ex_raw['Selected'], 'MIDCount'] = 0
            ex_raw.drop(columns=['Selected'], inplace=True)
            result_auto = ex_raw.copy()
            result_auto.to_csv(os.path.join(self.outpath, 'SpotGF_auto_threshold.gem'), sep='\t', index=False)
            if visualize:
                save_path = os.path.join(self.outpath, 'Spatial_automatic.png')
                adata = self.gem2adata(result_auto)
                self.expression_figure(adata, save_path, spot_size)

        # Proportion-based filtering
        if proportion is not None:
            print("Generate SpotGF-denoised data based on proportion")
            drop_pre = int(len(GF_df) / 10 * proportion)
            if drop_pre == 0:
                save_pro = GF_df
            else:
                cutoff_value = heapq.nlargest(drop_pre, GF_df['SpotGF_score'])[-1]
                save_pro = GF_df[GF_df['SpotGF_score'] >= cutoff_value]

            save_gene = list(save_pro.geneID)
            save = set(save_gene + gsave_high)
            ex_raw['Selected'] = ex_raw['geneID'].isin(save)
            ex_raw.loc[~ex_raw['Selected'], 'MIDCount'] = 0
            ex_raw.drop(columns=['Selected'], inplace=True)
            result = ex_raw.copy()
            result.to_csv(os.path.join(self.outpath, f'SpotGF_proportion_{proportion}.gem'), sep='\t', index=False)
            if visualize:
                save_path = os.path.join(self.outpath, f'Spatial_proportion_{proportion}.png')
                adata = self.gem2adata(result)
                self.expression_figure(adata, save_path, spot_size)
        else:
            result = ex_raw.copy()

        return result


########### Original Execution Code Starts Here #############

# Define directories and parameters
bench_data_dir = '/home/s2022310812/bench_data'  # [Modify] bench_data path
dropout_data_dir = '/home/s2022310812/dropout_data'  # [Modify] dropout_data path
datasets = [
    "IDC",
    # "COAD",
    # "READ",
    # "LYMPH_IDC",
]
model = "STAGATE"  # [Modify] Model name
percents = [
    '10%',
    # '20%',
    # '30%'
]
samples = [
    '1',
    # '2',
    # '3',
    # '4',
    # '5'
]

trial = '1'  # [Modify] Trial number (1-5)
man_sc = {}
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')  # [Modify] GPU device

for dataset in datasets:
    for percent in percents:
        for sample in samples:
            bench_path = os.path.join(bench_data_dir, dataset, 'adata')
            dropout_path = os.path.join(dropout_data_dir, dataset, f"{percent}-{sample}")

            adata_names = [f for f in os.listdir(bench_path) if os.path.isfile(os.path.join(bench_path, f))]

            for adata_name in adata_names:
                # Read and preprocess benchmark data
                adata = ad.read_h5ad(os.path.join(bench_path, adata_name))
                sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=False)
                sc.pp.log1p(adata)
                sc.pp.scale(adata)

                # Read and preprocess dropout data
                adata_drop = ad.read_h5ad(os.path.join(dropout_path, adata_name))
                sc.pp.normalize_total(adata_drop, target_sum=1e6, exclude_highly_expressed=False)
                sc.pp.log1p(adata_drop)
                sc.pp.scale(adata_drop)  # Fixed typo from 'scalse' to 'scale'

                # Define temporary paths
                temp_gem_path = os.path.join(bench_data_dir, dataset, f"temp_{percent}_{sample}_{adata_name}.gem")
                temp_out_path = os.path.join(bench_data_dir, dataset, f"output_{percent}_{sample}")

                # Extract dropped mask
                if 'dropped' in adata_drop.layers:
                    dropped_mask = torch.tensor(adata_drop.layers['dropped'], dtype=torch.bool, device=device)
                else:
                    print(f"'dropped' layer not found in {adata_drop}")
                    continue  # Skip if 'dropped' layer is missing

                # Convert dropout data to GEM format
                coords = adata_drop.obsm['spatial']
                genes = adata_drop.var_names
                original_genes = genes.tolist()
                original_cells = np.array(['{}_{}'.format(int(x), int(y)) for x, y in coords])

                X = adata_drop.X
                if isinstance(X, csr_matrix):
                    X = X.toarray()

                # Vectorized GEM conversion
                gene_ids, x_coords, y_coords = np.where(X > 0)
                counts = X[gene_ids, x_coords, y_coords]
                rows = [
                    [genes[j], int(coords[i][0]), int(coords[i][1]), int(X[i, j])]
                    for i, j in zip(x_coords, gene_ids)
                    if X[i, j] > 0
                ]
                gem_df = pd.DataFrame(rows, columns=['geneID', 'x', 'y', 'MIDCount'])
                gem_df.to_csv(temp_gem_path, sep='\t', index=False)

                # SpotGF parameter settings
                binsize = 10
                proportion = 0.5
                auto_threshold = True
                lower = 0
                upper = sys.float_info.max
                max_iterations = 10000
                visualize = False
                spot_size = 5
                alpha = 0

                # Apply SpotGF
                spotgf = SpotGF(temp_gem_path, binsize, proportion, auto_threshold, lower, upper,
                                max_iterations, temp_out_path, visualize, spot_size, alpha)
                GF_df = spotgf.calculate_GFscore(temp_gem_path, binsize, alpha, max_iterations, lower, upper)
                new_gem = spotgf.generate_GFgem(temp_gem_path, GF_df, proportion, auto_threshold, visualize, spot_size,
                                                all_genes=original_genes)

                # Convert new_gem back to adata2, ensuring original order
                new_gem['x_y'] = new_gem['x'].astype(str) + '_' + new_gem['y'].astype(str)
                wide = new_gem.pivot_table(index='x_y', columns='geneID', values='MIDCount', fill_value=0)

                # Ensure all original genes are present
                for g in original_genes:
                    if g not in wide.columns:
                        wide[g] = 0

                # Ensure all original cells are present
                for c in original_cells:
                    if c not in wide.index:
                        wide.loc[c] = 0

                # Reorder to match original cells and genes
                wide = wide.loc[original_cells, original_genes]

                obs = pd.DataFrame(index=original_cells)
                var = pd.DataFrame(index=original_genes)
                X2 = sparse.csr_matrix(wide.values)
                adata2 = ad.AnnData(X2, obs=obs, var=var)
                coords2 = np.array([list(map(int, idx.split('_'))) for idx in adata2.obs.index])
                adata2.obsm['spatial'] = coords2

                # Apply the same scaling to adata2
                sc.pp.scale(adata2)

                # Step 1: Convert adata.X to float32 before creating the PyTorch tensor
                if isinstance(adata.X, np.ndarray):
                    adata_X = adata.X.astype(np.float32)
                    adata_tensor = torch.tensor(adata_X, dtype=torch.float32, device=device)
                elif isinstance(adata.X, csr_matrix):
                    adata_X = adata.X.toarray().astype(np.float32)
                    adata_tensor = torch.tensor(adata_X, dtype=torch.float32, device=device)
                else:
                    print('Unsupported adata.X data format')
                    continue  # Skip if unsupported format

                # Step 2: Convert adata2.X to float32 before creating the PyTorch tensor
                if isinstance(adata2.X, np.ndarray):
                    adata2_X = adata2.X.astype(np.float32)
                    adata2_tensor = torch.tensor(adata2_X, dtype=torch.float32, device=device)
                elif isinstance(adata2.X, csr_matrix):
                    adata2_X = adata2.X.toarray().astype(np.float32)
                    adata2_tensor = torch.tensor(adata2_X, dtype=torch.float32, device=device)
                else:
                    print('Unsupported adata2.X data format')
                    continue  # Skip if unsupported format

                # Step 3: Get the indices of the dropped values (where the mask is True)
                dropped_indices = torch.nonzero(dropped_mask, as_tuple=True)

                # Step 4: Extract the corresponding values for dropped spots
                adata_values = adata_tensor[dropped_indices]
                adata2_values = adata2_tensor[dropped_indices]

                # Step 5: Convert to CPU and calculate Manhattan distance
                adata_values_cpu = adata_values.cpu().numpy()
                adata2_values_cpu = adata2_values.cpu().numpy()

                manhattan_distance = np.abs(adata_values_cpu - adata2_values_cpu).sum()
                num_values = adata_values_cpu.size
                manhattan_score = manhattan_distance / num_values

                # Store the result in the dictionary
                key = f"{dataset}/{adata_name}"
                man_sc[key] = manhattan_score

                print(f"Manhattan score for {key}: {manhattan_score}")

            # Prepare to save results
            row_name = f"Trial_{trial}"
            my_dir = os.path.join('/home/s2022310812/bench_results', model, dataset)
            csv_path = os.path.join(my_dir, f"{percent}-{sample}_result.csv")
            os.makedirs(my_dir, exist_ok=True)

            # Extract column names and values
            columns = [key.split("/")[-1].split(".")[0] for key in man_sc.keys()]
            values = list(man_sc.values())

            # Create a DataFrame for the new row
            data = pd.DataFrame([values], columns=columns, index=[row_name])

            # Check if the file exists and append accordingly
            if os.path.exists(csv_path):
                existing_data = pd.read_csv(csv_path, index_col=0)
                updated_data = pd.concat([existing_data, data])
            else:
                updated_data = data

            # Save back to CSV
            updated_data.to_csv(csv_path)  # [Modify] Ensure this line is active unless debugging

            print(f"Saved results to {csv_path}")
