"""
This script will extract two intensity features, median and std, as well as six haralick features from image.
It is tested to work with Visium he and if images.
A conda env is available in the project folder. 

# input folder must have the raw tif image and the spot_metadata csv file.
# The spot metadata file MUST have 'X', 'Y', and 'Spor_radius' columns

# example usage:
conda activate /project/shared/xiao_wang/projects/MOCCA/conda_env
python /project/shared/xiao_wang/projects/MOCCA/code/feature_extraction.py \
    /project/shared/xiao_wang/projects/MOCCA/data/Visium/ovarian_cancer_immune \
    if \
    /project/shared/xiao_wang/projects/MOCCA/data/Visium/ovarian_cancer_immune/intermediate \
    
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
from skimage import io, img_as_float32, morphology, exposure
from skimage.feature import greycomatrix, greycoprops
from itertools import product
from sklearn.preprocessing import minmax_scale
from skimage.color import separate_stains, hdx_from_rgb

from cucim import CuImage
import cupy as cp
from cucim.skimage.color import separate_stains
from cucim.skimage.exposure import equalize_adapthist
from cuml.decomposition import PCA  # GPU-accelerated PCA
from cuml.manifold import UMAP




def extract_img_features(
    input_path,
    input_type,
    output_path,
    img=None,
    img_meta=None,
    feature_mask_shape="spot",
):
    """
    Extract features from image. Works with IF or HE image from Visium tif files.
    For block feature, a square will be drawn around each spot. Since it is bigger than 
    the spot itself, it is more suitable to extract texture features. 
    For Spot feature, only area in the actual sequencing spot will be uses. 
    It is more suitable to extract intensity features.

    Parameters
    ----------
    input_path : str
        input folder containing all necessary files. 
    input_type : str
        input image type, select from {'if','he'}.
    output_path : str
        output folder path.
    img : None or np.array, optional
        alternative input for image, will override input_path.
    img_meta : None or np.array, optional
        alternative input for image metadata, will override input_path.
    feature_mask_shape : {'spot', 'block'}
        type of feature extracted.
    """
    intensity_fn = os.path.join(
        os.path.abspath(output_path),
        "{}_level_texture_features.csv".format(feature_mask_shape)
    )
    texture_fn = os.path.join(
        os.path.abspath(output_path),
        "{}_level_intensity_features.csv".format(feature_mask_shape)
        )
    if (os.path.exists(intensity_fn)) == (os.path.exists(texture_fn)) == True:
        print('Features are already extracted.')
        return
    if img_meta is None:
        img_meta = pd.read_csv(
            os.path.join(input_path,"Spot_metadata.csv"), index_col=0)
    if img is None:
        img_tif = [x for x in os.listdir(input_path) if "tif" in x][0]
        img_tif = os.path.join(input_path, img_tif)
        if input_type == "if":
            # the indexing is a workaround for the strange Visium if image channels.
            #img = io.imread(img_tif)
            #img = img_as_float32(img)
            #img = (255 * img).astype("uint8")
            print('if')
        else:
            img = io.imread(img_tif)
            print(img.shape)

            row = img_meta.iloc[0]
            r = row["Spot_radius"].astype(int)
            print(r)
            img_c = np.pad(img, ((0, 2*r), (0, 2*r), (0, 0)), mode='constant', constant_values=0)
            
            device_id = 2  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
            with cp.cuda.Device(device_id):
                #img_c = io.imread(img_tif)

                img = cp.asarray(img_c)

                # Define the patch size (you can adjust this depending on the available GPU memory)
                PATCH_SIZE = (1024, 1024)  # Adjust based on memory constraints

                # Function to process a patch
                def process_patch(patch):
                    patch = separate_stains(patch, hdx_from_rgb)

                    # Min-max scale using CuPy
                    patch = (patch - cp.min(patch)) / (cp.max(patch) - cp.min(patch))
                    patch = cp.clip(patch, 0, 1)

                    # Adaptive histogram equalization
                    patch = equalize_adapthist(patch, clip_limit=0.01)

                    # Convert to uint8 format
                    return (255 * patch).astype(cp.uint8)

                # Function to batch process the image
                def batch_process_image(img, patch_size):
                    

                    # Get the image dimensions
                    img_height, img_width = img.shape[:2]

                    # Create an empty array to store the processed image
                    processed_img = cp.zeros_like(img, dtype=cp.uint8)

                    # Iterate over the image in patches
                    for i in range(0, img_height, patch_size[0]):
                        for j in range(0, img_width, patch_size[1]):
                            # Define the patch
                            patch = img[i:i + patch_size[0], j:j + patch_size[1]]

                            # Process the patch and store it in the result
                            processed_patch = process_patch(patch)
                            processed_img[i:i + patch_size[0], j:j + patch_size[1]] = processed_patch

                    return processed_img

                # Call the batch processing function
                img_send = batch_process_image(img, PATCH_SIZE)
                
            """
            img = io.imread(img_tif)
            # normalize image with color deconv
            print('Normalizing image...')
            img = separate_stains(img, hdx_from_rgb)
            img = minmax_scale(img.reshape(-1, 3)).reshape(img.shape)
            img = np.clip(img, 0, 1)
            img = exposure.equalize_adapthist(img, clip_limit=0.01)
            img = (255 * img).astype("uint8")
            """
    # Hard coded type of Haralick features and Angles for searching for neighboring pixels
    # hard coded number of angles to be 4, meaning horizontal, vertical and two diagonal directions.
    # extracting block shaped features
    if feature_mask_shape == "block":
        tmp = img_meta.sort_values(["Row", "Col"])
        block_y = int(np.median(tmp.Y.values[2:-1] - tmp.Y.values[1:-2]) // 2)
        tmp = img_meta.sort_values(["Col", "Row"])
        block_x = int(np.median(tmp.X.values[2:-1] - tmp.X.values[1:-2]) // 2)
        block_r = min(block_x, block_y)
        block_x = block_y = block_r
    print("Prossessing {}".format(input_path))
    feature_set = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation",
    ]
    text_features = []
    intensity_features = []
    device_id=3
    with cp.cuda.Device(device_id):
        img = cp.asarray(img_send)
        print (img.shape)
        for i in range(img_meta.shape[0]):
            if (i + 1) % 100 == 0:
                print("Processing {} spot out of {} spots".format(i + 1, img_meta.shape[0]))
            row = img_meta.iloc[i]
            x, y, r = row[["X", "Y", "Spot_radius"]].astype(int)
            if feature_mask_shape == "spot":
                
                spot_img = img[x - r : x + r + 1, y - r : y + r + 1]
                spot_mask = morphology.disk(r)
                if spot_img.shape[:2] != spot_mask.shape[:2]:
                    print(f"Dimensions don't match for x={x}, y={y}.")
                    print(f"spot_img shape: {spot_img.shape}, spot_mask shape: {spot_mask.shape}")
                # only use the spot, not the bbox

                spot_img = cp.einsum("ij,ijk->ijk", spot_mask, spot_img)
            else:
                spot_img = img[x - block_x : x + block_x + 1, y - block_y : y + block_y + 1]
                spot_mask = np.ones_like(spot_img[:, :, 0], dtype="bool")

            # extract texture features
            ith_texture_f = []
            for c in range(img.shape[2]):
                glcm = cp.array(greycomatrix(
                    spot_img[:, :, c].get(),
                    distances=[1],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=256,
                    symmetric=True,
                    normed=False,
                ))
                glcm_sum = cp.sum(glcm, axis=(0, 1))  # Sum of GLCM values on GPU
                glcm = glcm / cp.maximum(glcm_sum, 1e-10)  # Normalize with GPU operation
                for feature_name in feature_set:
                    ith_texture_f += greycoprops(glcm.get(), feature_name)[0].tolist()
            # The first 6 features are intensity features, and the rest are Haralicks.
            text_features.append(ith_texture_f)



            # extract intensity features
            int_low = 0.2
            int_high = 0.8
            int_step = 0.1
            q_bins = np.arange(int_low, int_high, int_step)
            ith_int_f = []
            for c in range(img.shape[2]):
                for t in q_bins:
                    ith_int_f.append(cp.percentile(spot_img[:, :, c][spot_mask == True], t))
            intensity_features.append(ith_int_f)

        # Naming the features. f stands for channels, A stands for angles.
        # construct texture feature table
        channels = ["f" + str(i) for i in range(img.shape[2])]
        col_names = product(channels, feature_set, ["A1", "A2", "A3", "A4"])
        col_names = ["_".join(x) for x in col_names]
        text_features = pd.DataFrame(text_features, index=img_meta.index, columns=col_names)
        # construct intensity feature table
        intensity_features = pd.DataFrame(
            intensity_features,
            index=img_meta.index,
            columns=[
                "_".join(x) for x in product(channels, ["{:.1f}".format(x) for x in q_bins])
            ],
        )
        
        # Transfer back to CPU before saving as CSV
        text_features = cp.asnumpy(text_features)
        intensity_features = cp.asnumpy(intensity_features)

        # Convert the NumPy arrays back to DataFrames
        text_features = pd.DataFrame(text_features)
        intensity_features = pd.DataFrame(intensity_features)

        # Overwrite the first column with the Spot_IDs
        text_features.iloc[:, 0] = img_meta.index.values
        intensity_features.iloc[:, 0] = img_meta.index.values

        # Save the data to CSV
        text_features.to_csv(texture_fn, index=False)
        intensity_features.to_csv(intensity_fn, index=False)
    return text_features, intensity_features

def preprocess_features(IF, method="PCA", n_components=-1):
    """
    Preprocess the image features with PCA or UMAP and overwrite the original file.

    Parameters:
        IF_path (str): Path to the image features file (input and overwritten).
        method (str): Preprocessing method, either "PCA" or "UMAP".
        n_components (int): Number of components to keep.
    """
    device_id = 3  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
    with cp.cuda.Device(device_id):
        # Load the feature matrix (assuming it's stored as a NumPy array)
        IF = cp.array(IF)

        # Center and scale the features
        IF = (IF - cp.mean(IF, axis=0)) / cp.std(IF, axis=0)
        if (n_components == -1):
            n_components = IF.shape[1]
        if method == "PCA":
            print("Preprocessing with PCA...")
            pca = PCA(n_components=n_components)
            IF_preprocessed = pca.fit_transform(IF)
        elif method == "UMAP":
            print("Preprocessing with UMAP...")
            umap = UMAP(n_components=n_components)
            IF_preprocessed = umap.fit_transform(IF)
        else:
            raise ValueError("Invalid method! Choose 'PCA' or 'UMAP'.")

        print(f"Preprocessed features saved")
        return IF_preprocessed

def calculate_spot_dist_gpu(n, sigma_n, euc_dist2):
    """
    Calculate the spot distance based on Gaussian kernel regulated Euclidean distance.
    n: Index of the target row (spot).
    sigma_n: Precomputed vector of sigma values for all rows.
    euc_dist2: Precomputed pairwise squared Euclidean distances.
    
    Returns a vector of probabilities (p_n_n2) for all spots.
    """
    device_id = 3  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
    with cp.cuda.Device(device_id):
        tmp = -euc_dist2[n] / (2 * sigma_n[n] ** 2)
        tmp = tmp - cp.max(cp.delete(tmp, n))  # Normalize by subtracting max, excluding n
        d_n_n2 = cp.exp(tmp)
        d_n_n2[n] = 0  # Set self-probability to 0
        denominator = cp.sum(d_n_n2)
        return d_n_n2 / denominator if denominator != 0 else cp.zeros_like(d_n_n2)

def compute_p_nn_tsne(IF, U, margin, power_tsne_factor):
    device_id = 3  # Specify GPU device
    with cp.cuda.Device(device_id):
        # Step 1: Compute pairwise squared Euclidean distances
        euc_dist2 = cp.square(cp.linalg.norm(IF[:, None, :] - IF[None, :, :], axis=2))
        
        # Step 2: Calculate sigma_n values for all spots
        N = IF.shape[0]
        sigma_n = cp.zeros(N)
        for i in range(N):
            distances = cp.sort(euc_dist2[i])  # Sort distances for margin calculation
            sigma_n[i] = distances[min(int(U * N), N - 1)] + margin  # Adjust index if U*N exceeds N
        
        # Step 3: Calculate p_n_n for each spot
        p_n_n = cp.zeros((N, N), dtype=cp.float32)
        for n in range(N):
            p_n_n[n] = calculate_spot_dist_gpu(n, sigma_n, euc_dist2)
        
        # Step 4: Symmetrize the matrix
        p_nn_tsne = 1 - (p_n_n + p_n_n.T) / 2
        
        # Step 5: Apply power transformation
        p_nn_tsne = cp.power(p_nn_tsne, N / power_tsne_factor)
        
        # Check for NaN values
        if cp.any(cp.isnan(p_nn_tsne)):
            raise ValueError("Error: Latent space proximity matrix contains NaN!")
        
        return p_nn_tsne

def fr(par, data):
    """
    Function to calculate the cost given parameters and data.
    
    Args:
        par (cp.ndarray): Flattened parameter matrix (alpha).
        data (tuple): A tuple containing:
            - lambda_matrix (cp.ndarray): Lambda matrix.
            - p_nn_tsne (cp.ndarray): Proximity matrix from t-SNE.
            - proximity (cp.ndarray): Input proximity matrix.
            - k (float): A scalar constant.

    Returns:
        float: The calculated cost.
    """
    device_id = 3  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
    with cp.cuda.Device(device_id):
        # Unpack the data
        lambda_matrix, p_nn_tsne, proximity, k = data
        
        # Reshape the parameter vector into the alpha matrix
        alpha = par.reshape(proximity.shape)
        alpha = alpha + alpha.T  # Ensure symmetry
        alpha = cp.array(alpha)  # Convert to cupy if it's a numpy array

        # Q = I + 4*diag(rowSums(alpha)) - 4*alpha
        I = cp.eye(alpha.shape[0])  # Identity matrix
        diag_sum_alpha = cp.diag(cp.sum(alpha, axis=1))  # Diagonal matrix of row sums



        Q = I + 4 * diag_sum_alpha - 4 * alpha
        Q += cp.eye(Q.shape[0]) * 1e-8  # Regularization to avoid numerical issues
        
        # Compute the determinant of Q (logarithmically for numerical stability)
        det_log = cp.linalg.slogdet(Q)[1]  # Compute log(det(Q))
        
        # Compute the cost
        term1 = cp.sum(alpha * lambda_matrix * p_nn_tsne)
        term2 = (k / 2) * det_log
        cost = term1 - term2
        # Check for NaN or Inf in the result
        if cp.isnan(cost).any() or cp.isinf(cost).any():
            raise ValueError("NaN or Inf detected in cost")
        # Convert to a scalar and return (if using GPU arrays)
        return cost.item()
def gr(par, data):
    """
    Function to calculate the gradient given parameters and data.
    
    Args:
        par (cp.ndarray): Flattened parameter matrix (alpha).
        data (tuple): A tuple containing:
            - lambda_matrix (cp.ndarray): Lambda matrix.
            - p_nn_tsne (cp.ndarray): Proximity matrix from t-SNE.
            - proximity (cp.ndarray): Input proximity matrix.
            - k (float): A scalar constant.

    Returns:
        cp.ndarray: The gradient as a matrix.
    """
    device_id = 3  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
    with cp.cuda.Device(device_id):
        # Unpack the data
        lambda_matrix, p_nn_tsne, proximity, k = data
        
        # Reshape the parameter vector into the alpha matrix
        alpha = par.reshape(proximity.shape)
        alpha = alpha + alpha.T  # Ensure symmetry
        alpha = cp.array(alpha)  # Convert to cupy if it's a numpy array

        # Q = I + 4*diag(rowSums(alpha)) - 4*alpha
        I = cp.eye(alpha.shape[0])  # Identity matrix
        diag_sum_alpha = cp.diag(cp.sum(alpha, axis=1))  # Diagonal matrix of row sums
        Q = I + 4 * diag_sum_alpha - 4 * alpha
        Q += cp.eye(Q.shape[0]) * 1e-5  # Regularization to stabilize

        # Q^-1 (Inverse of Q)
        Q_inv = cp.linalg.inv(Q)
        Q_inv = (Q_inv + Q_inv.T) / 2  # Symmetrize for numerical stability
        
        # Extract the diagonal elements of Q_inv into a temporary matrix
        tmp = Q_inv.copy()
        tmp[:, :] = cp.diag(Q_inv)[:, None]  # Broadcast diagonal elements across rows

        # Compute the gradient
        gradient = lambda_matrix * p_nn_tsne - 2 * k * (tmp + tmp.T - 2 * Q_inv)
        
        if cp.isnan(gradient).any() or cp.isinf(gradient).any():
            raise ValueError("NaN or Inf detected in gradient")

        return gradient.flatten().get()  # Return as a flattened array for optimization routines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract features from image. Works with IF or HE image from Visium tif files.\
            For block feature, a square will be drawn around each spot. Since it is bigger than \
            the spot itself, it is more suitable to extract texture features. \
            For Spot feature, only area in the actual sequencing spot will be uses. \
                It is more suitable to extract intensity features.",
    )

    parser.add_argument(
        "input_path", type=str, help="Input folder containing all necessary files."
    )
    parser.add_argument(
        "input_type", type=str, help="Input image type, select from {'if','he'}."
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="./intermediate",
        help="Output folder path.",
    )
    parser.add_argument(
        "--feature_mask_shape",
        "-m",
        type=str,
        default="spot",
        help="Type of feature extracted. {'spot', 'block'}",
    )

    args = parser.parse_args()
    # Todo: decide if checking metadata function should be added, or force the user to provide the correct format.
    input_path = os.path.abspath(args.input_path)
    input_type = args.input_type
    output_path = os.path.abspath(args.output_path)
    feature_mask_shape = args.feature_mask_shape

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Processing {}".format(input_path))
    output_fn = output_path + "/{}_level_texture_features.csv".format(
        feature_mask_shape
    )
    if os.path.exists(output_fn):
        print("Image feature already extracted. Stop operation.")
    else:
        _ = extract_img_features(
            input_path, input_type, output_path,
            feature_mask_shape=feature_mask_shape)

# Debugging codes
# f_t, f_i = extract_img_features(
#     '','HE','',img = center_patch_sep, img_meta = center_patch_meta,
#     feature_mask_shape='spot')

# # valid_cols = [x for x in f_block.columns if 'homo' not in x]
# # valid_cols = [x for x in valid_cols if 'corr' not in x]
# # f_pca_b = PCA(n_components=10).fit_transform(scale(f_block[valid_cols]))
# # clusters_b = [str(x) for x in k_means(f_pca_b,5)[1]]

# f_pca_b = PCA(n_components=10).fit_transform(scale(f_i))
# clusters_b = [str(x) for x in k_means(f_pca_b,3)[1]]

# _ = sns.mpl.pyplot.figure(figsize = (16,16))
# ax = sns.scatterplot(
#     y = center_patch_meta.X, x = center_patch_meta.Y,
#     hue=clusters_b, hue_order = sorted(set(clusters_b)), markers = ['h'],
#     linewidth=0, alpha=0.25, s=250)
# ax.set_facecolor('grey')
# io.imshow(center_patch[:,:,0], alpha=0.9)
# io.imshow(center_patch_sep[:,:,0],cmap='gray')