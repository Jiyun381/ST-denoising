"""
A convinient function to wrap all necessary processed in Sprod. Minimal data processing is required.
A 'Counts.txt' and a 'Spot_metadata.csv' is required to run this script.
Also, if the input folder contains a image tif file, it will be used as the input matching
image for feature extraction. Both files have rows as spots and with the same order.
For data with matching image, the "Spot_metadata.csv" must have a "Spot_radius"
column for spot features, or "Row" and "Col" columns for block features.

Dependency
----------
R >= 4.0.2
    distances
    dplyr

Python dependencies will be handeled automatically by installing the sprod package.

"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sprod.feature_extraction import extract_img_features, preprocess_features, compute_p_nn_tsne, gr, fr
from sprod.pseudo_image_gen import make_pseudo_img
from sprod.slide_seq_stiching import stiching_subsampled_patches
from sprod.slideseq_make_patches import subsample_patches
import subprocess
import platform
import cupy as cp
from scipy.optimize import minimize
from scipy.linalg import solve, eigh


def sprod_worker(cmd):
    """
    A simple wrapper for running denoise jobs.
    --------
    denoise.R options for reference.
    make_option(c("-e", "--Exp"), help="Expression matrix"),
    make_option(c("-c", "--Cspot"), help="Spot metadata, contains X, Y coordinates."),
    make_option(c("-f", "--ImageFeature"), help="Extracted image features"),
    make_option(c("-n", "--numberPC"), help="# Number of PCs to use"),
    make_option(c("-r", "--Rratio"),help="Spot neighborhood radius ratio, 0-1, radius=R*min(xmax-xmin,ymax-ymin)"),
    make_option(c("-u", "--U"),help="# perplexity, Tunable"),
    make_option(c("-k", "--K"),help="latent space dimension, Tunable"),
    make_option(c("-l", "--lambda"),help="# regularizer, tunable"),
    make_option(c("-t", "--L_E"),help="regularizer for denoising"),
    make_option(c("-m", "--margin"), help="Margin for bisection search, smaller = slower => accuracy u"),
    make_option(c("-d","--diagNeighbor"),help = "To prevent over smooth."),
    make_option(c("-s", "--scriptPath"), help="path to the software folder"),
    make_option(c("-o", "--outputPath"), help="Output path"),
    make_option(c("-p", "--projectID"), help="# project name, First part of names in the output"),
    """
    log_fn = cmd[-1]
    cmd = cmd[:-1]
    print(cmd)
    with open(log_fn, "a") as outfile:
        subprocess.run(cmd, stdout=outfile, stderr=outfile)
    # os.system('rm -r output_fn') # Get rid of output.
    return


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main function for running sprod.",
    )

    parser.add_argument(
        "input_path", 
        type=str, 
        help="Input folder containing all necessary files. Must have Counts.txt \
            and Spot_metadata.csv."
    )

    parser.add_argument("output_path", type=str, help="Output path")

    parser.add_argument(
        "--input_type",
        "-y",
        default="single",
        type=str,
        help="The input type decides the running mode for sprod, select from {'single','batch'}. \
            For smaller datasets, such as Visium, use 'single' mode. For larger datasets such as \
            Slide-seq, use 'batch' mode",
    )

    parser.add_argument(
        "--output_prefix",
        "-p",
        default="sprod",
        type=str,
        help="Output prefix used in the output.",
    )

    parser.add_argument(
        "--sprod_npc",
        "-sn",
        type=int,
        default=-1,
        help="Number of PCs to use, positive integers. -1 to use all PCs from the features.",
    )

    parser.add_argument(
        "--sprod_umap",
        "-su",
        default=False,
        action="store_true",
        help="Toggle to use UMAP on top of PCA to represent features. ",
    )

    parser.add_argument(
        "--sprod_R",
        "-r",
        default=0.08,
        type=float,
        help="Spot neighborhood radius ratio, 0-1, \
            radius=R*min(xmax-xmin,ymax-ymin).",
    )

    parser.add_argument(
        "--spord_perplexity",
        "-u",
        default=250,
        type=int,
        help="Perplexity, used in Sprod to find the proper Gaussian kernal \
            for distant representation.",
    )

    parser.add_argument(
        "--sprod_margin",
        "-g",
        default=0.001,
        type=float,
        help="Margin for bisection search, used in Sprod to find the proper \
            Gaussian kernal. smaller = slower => accurate u",
    )

    parser.add_argument(
        "--sprod_latent_dim",
        "-k",
        default=10,
        type=int,
        help="Dimension of the latent space used in sprod to represent spots.",
    )

    parser.add_argument(
        "--sprod_graph_reg",
        "-l",
        default=1,
        type=float,
        help="Regularization term for spot graph contructed in sprod.",
    ),

    parser.add_argument(
        "--sprod_weight_reg",
        "-w",
        default=0.4,
        type=float,
        help="Regularization term for the denoising weights.",
    )

    parser.add_argument(
        "--sprod_diag",
        "-d",
        action="store_true",
        default=False,
        help="Toggle to force graph weights to be diagnoal, useful in reducing \
            over smoothing",
    )

    parser.add_argument(
        "--image_feature_type",
        "-i",
        type=str,
        default="spot_intensity",
        help="Type of feature extracted. combination from {'spot', 'block'} and \
            {'intensity', 'texture'} with '_' as delimiter. Only relevant if the \
            input dataset contains an matching tif image.",
    )

    parser.add_argument(
        "--warm_start",
        "-ws",
        default=False,
        action="store_true",
        help="Toggle for warm start, which will skip all preprocessing steps \
            including feature extraction and all prep steps in the batch mode.",
    )

    parser.add_argument(
        "--diag_mode",
        "-dg",
        default=False,
        action="store_true",
        help="Toggle invoking plotting diagnosis plots in denoise.R",
    )

    parser.add_argument(
        "--num_of_batches",
        "-pb",
        type=int,
        default=10,
        help="How many times subsampling is ran. Only works when --input_type is 'batch'.",
    )

    parser.add_argument(
        "-ci",
        "--img_type",
        type=str,
        default="he",
        help="Input image type. {'he', 'if'}. The 'if' mode is only tested on Visium-assocaited data.",
    )

    parser.add_argument(
        "--custom_feature",
        "-cf",
        default=None,
        help="Option for advanced users. A custom spot by feature csv file can be used \
            together with sprod extracted features. Must have matching spot names and order.\
                The rows names should be spot names and the columns should be feature names.",
    )

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    input_type = args.input_type
    warm_start = args.warm_start
    output_prefix = args.output_prefix
    sprod_npc = args.sprod_npc
    sprod_umap = args.sprod_umap
    sprod_R = args.sprod_R
    spord_perplexity = args.spord_perplexity
    sprod_margin = args.sprod_margin
    sprod_latent_dim = args.sprod_latent_dim
    sprod_graph_reg = args.sprod_graph_reg
    sprod_weight_reg = args.sprod_weight_reg
    sprod_diag = args.sprod_diag
    sprod_diagnose = args.diag_mode
    image_feature_type = args.image_feature_type
    img_type = args.img_type
    pb = args.num_of_batches
    custom_features_fn = args.custom_feature
    os_type = platform.system()

    # getting script path for supporting codes.
    sprod_path = os.path.abspath(__file__)
    sprod_path = sprod_path.replace("sprod.py", "sprod")
    sprod_script = os.path.join(sprod_path, "denoise.R")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    # Setting up logs
    log_fn = os.path.join(output_path, "sprod_log.txt")
    if os.path.exists(log_fn):
        os.remove(log_fn)
    logging.basicConfig(
        filename=log_fn,
        format="%(asctime)s,%(levelname)s:::%(message)s",
        datefmt="%H:%M:%S",
        level="INFO",
    )
    # redirects stdout and stderr to logger
    stdout_logger = logging.getLogger("STDOUT")
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger("STDERR")
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    feature_fn = os.path.join(
            input_path,
            "{}_level_{}_features.csv".format(*image_feature_type.split("_")),
        )

    # cold start means image features will be extracted first.
    if not warm_start:
        logging.info("Cold start, processing from counts and image data.")
        # check if input path contains necessary data
        if not (
            os.path.exists(os.path.join(input_path, "Counts.txt"))
            == os.path.exists(os.path.join(input_path, "Spot_metadata.csv"))
            == True
        ):
            logging.error(
                "Counts.txt and/or Spotmeta.csv is missing from {}".format(input_path)
            )
            raise ValueError()

        # check if the input path have an tif image, if not, make a pseudo image.
        num_tifs = len([x for x in os.listdir(input_path) if x[-4:] == ".tif"])
        if num_tifs == 1:
            logging.info(
                "Extracting intensity and texture features from matching {} image".format(
                    img_type.upper()
                )
            )
            _ = extract_img_features(input_path, img_type, input_path)
        elif num_tifs == 0:
            logging.info("Use spot cluster probability as pseudo image features")
            cts_fn = os.path.join(input_path, "Counts.txt")
            spots_fn = os.path.join(input_path, "Spot_metadata.csv")
            dp_script_path = os.path.join(sprod_path, "dirichlet_process_clustering.R")
            make_pseudo_img(cts_fn, spots_fn, input_path, "hdbscan", dp_script_path)
            pseudo_img_feature_fn = os.path.join(
                input_path, "pseudo_image_features.csv"
            )
        else:
            logging.error(
                "More than one images are present. Please remove the unwanted ones."
            )
            raise ValueError()

        # When the input_type is batch, will run subsampling process.
        if input_type == "batch":
            logging.info("Making subsamples from counts data.")
            intermediate_path = os.path.join(output_path, "intermediate")
            feature_fn = os.path.join(
                input_path,
                "{}_level_{}_features.csv".format(*image_feature_type.split("_")),
            )
            if not os.path.exists(feature_fn):
                if custom_features_fn is not None:
                    feature_fn = custom_features_fn
                    print(
                        "Using custom features."
                    )
                elif os.path.exists(pseudo_img_feature_fn):
                    feature_fn = pseudo_img_feature_fn
                    print(
                        "Image derived features not found, will use pseudo image features."
                    )
                else:
                    raise FileNotFoundError('Input features cannot found!')
                
            if not os.path.exists(intermediate_path):
                os.makedirs(intermediate_path)
                
            cts_fn = os.path.join(input_path, "Counts.txt")
            n_spots = sum(1 for _ in open(cts_fn))
            pn = max(2,int(np.ceil(n_spots / 5000)))
            if not os.path.exists(os.path.join(intermediate_path, "Batch_0_patch_0_Counts.txt")):
                subsample_patches(input_path, intermediate_path, feature_fn, pn, pb)
            input_path = intermediate_path
        elif input_type == "single":
            pass
        else:
            logging.error("Input type must be single or batch")
            raise ValueError()

    # Sprod denoising
    logging.info("Proceed to Sprod denoising.")
    if input_type == "batch":
        logging.info("Sprod-ready data format is batch")
        counts = [x for x in os.listdir(input_path) if "Counts" in x]
        patches = [x.replace("_Counts.txt", "") for x in counts]
        inputs = []
        for j, patch in enumerate(sorted(patches)):
            cts_fn = os.path.join(input_path, patch + "_Counts.txt")
            meta_fn = os.path.join(input_path, patch + "_Spot_metadata.csv")
            feature_fn = os.path.join(input_path, patch + "_F.csv")
            #pnn_fn = os.path.join(input_path, patch + '_pnn.csv')

            if not (
                os.path.exists(cts_fn)
                == os.path.exists(meta_fn)
                == os.path.exists(feature_fn)
                == True
            ):
                logging.error(
                    "At least one subsample failed. Please check the output folder."
                )
                raise ValueError()
            inputs.append([cts_fn, meta_fn, feature_fn])
    else:
        logging.info("Sprod-ready data format is single.")
        cts_fn = os.path.join(input_path, "Counts.txt")
        meta_fn = os.path.join(input_path, "Spot_metadata.csv")
        feature_fn = os.path.join(
            input_path,
            "{}_level_{}_features.csv".format(*image_feature_type.split("_")),
        )

        # always tries to use image derived features, then pseudo images.
        if not os.path.exists(feature_fn):
            pseudo_img_feature_fn = os.path.join(
                input_path, "pseudo_image_features.csv"
            )
            if custom_features_fn is not None:
                feature_fn = custom_features_fn
                print("Using custom features.")
            elif os.path.exists(pseudo_img_feature_fn):
                feature_fn = pseudo_img_feature_fn
                logging.info(
                    "Image derived features not found, will use pseudo image features."
                )
            else:
                raise FileNotFoundError("Image features not found.")
        else:
            logging.info("Image derived features found, will use these.")

        # Incorporate custom features
        if custom_features_fn is not None:
            tmp_features = pd.read_csv(feature_fn, index_col=0)
            custome_features = pd.read_csv(custom_features_fn, index_col=0)
            assert (
                tmp_features.index == custome_features.index
            ).all(), "Spots in custom feature matrix does not match those in exacted features."
            pooled_features = pd.concat([tmp_features, custome_features], axis=1)
            logging.info("Appended custom features into existing features.")
            feature_fn = os.path.join(input_path, "pooled_features.csv")
            pooled_features.to_csv(feature_fn)

        # final check for all required input files.
        if not (
            os.path.exists(cts_fn)
            == os.path.exists(meta_fn)
            == os.path.exists(feature_fn)
            == True
        ):
            raise ValueError("Required data is missing! from {}".format(input_path))
        inputs = [[cts_fn, meta_fn, feature_fn]]

    logging.info("Total number of sprod jobs : {}".format(len(inputs)))
    device_id = 3  # Change this to the ID of the GPU you want to use (e.g., 0, 1, etc.)
    with cp.cuda.Device(device_id):


        # Starting  denoise
        list_sprod_cmds = []
        for sprod_job in inputs:
                        # File names


            R_ratio = 0.08
            K=10
            L_E=0.4
            LAMBDA=1
            d= False
            cts_fn, meta_fn, feature_fn = [os.path.abspath(x) for x in sprod_job]
            if input_type == "batch":
                patch_name = cts_fn.split(os.sep)[-1].replace("_Counts.txt", "")
            else:
                patch_name = output_prefix
            
            project_name = patch_name
            E_denoised_fn = f"{project_name}_Denoised_matrix.txt"


            # Saving to files
            output_path= os.path.abspath(output_path)

            if os.path.exists(os.path.join(output_path, E_denoised_fn)) :
                print('skipping '  + project_name)
                continue
            # Load the counts data
            E = pd.read_csv(cts_fn, sep="\t", header=None, skiprows=1).iloc[:, 1:]
            print('Readingrownames')

            # Read the CSV file, assuming the row names are stored in the first column
            data = pd.read_csv(cts_fn,  sep="\t", index_col=0, header=0)
            #data.index = data.index.str.replace('"', '').str.strip()
            # Extract the row names (index)
            rownames_E = data.index.tolist()
            print(rownames_E[:3])


            print(E.shape)
            # Load the spot metadata
            C = pd.read_csv(meta_fn, index_col=0, header=0)
            
            # Check for Z column and calculate R
            if "Z" in C.columns:
                print("Z column is detected!\n")
                R = min(
                    C["X"].max() - C["X"].min(),
                    C["Y"].max() - C["Y"].min(),
                    C["Z"].max() - C["Z"].min()
                ) * R_ratio
            elif "X" in C.columns and "Y" in C.columns:
                print("XY columns are detected but not Z!\n")
                R = min(
                    C["X"].max() - C["X"].min(),
                    C["Y"].max() - C["Y"].min()
                ) * R_ratio
            else:
                raise ValueError(
                    "No correct columns are detected! Please check the column names of spot_metadata!"
                )
            
            # Determine file format and load image features
            file_ext = os.path.splitext(feature_fn)[-1].lower()
            if file_ext == '.csv':
                IF = pd.read_csv(feature_fn, index_col=0, header=0).values
            else:
                IF = pd.read_csv(feature_fn, index_col=0, header=0, sep='\t').values

            print("Constructing latent graph based on position and image features...\n")

                    
            # Determine whether "Z" exists and calculate distances
            #if C.shape[1] == 3:  # Assume columns ["X", "Y", "Z"]
            #    proximity_matrix = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2) <= R
            #else:  # Assume columns ["X", "Y"]
            # Extract the 'X' and 'Y' columns from the DataFrame and transfer them to the GPU
            coordinates = cp.asarray(C[['X', 'Y']].values)  # Convert to a cupy array (GPU)

            # Compute pairwise Euclidean distances on the GPU
            # Compute pairwise Euclidean distances using broadcasting
            # dist_matrix[i, j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
            diff = coordinates[:, cp.newaxis, :] - coordinates[cp.newaxis, :, :]
            dist_matrix = cp.linalg.norm(diff, axis=2)

            # Create a boolean matrix where distances are less than or equal to R
            proximity_matrix = dist_matrix <= R

            # Set diagonal of proximity matrix to False (no self-proximity)
            np.fill_diagonal(proximity_matrix, False)

            # Ensure alpha_n1_n2 == alpha_n2_n1 by making lower triangle of P False
            proximity_matrix[np.tril_indices_from(proximity_matrix)] = False

            # Display the proximity matrix summary
            P = proximity_matrix

            # Initialize ALPHA matrix
            ALPHA = 0.5 * P  # Simply multiply P by 0.5 to achieve the same result


            IF = preprocess_features(IF, method="PCA")
            p_nn_tsne = compute_p_nn_tsne(IF, U=250, margin=0.001, power_tsne_factor=7)
            print('Performing optimization')
            while True:
                # Perform optimization
                result = minimize(
                    fun=lambda x: fr(x, (LAMBDA, p_nn_tsne, P, K)),
                    jac=lambda x: gr(x, (LAMBDA, p_nn_tsne, P, K)),
                    x0=ALPHA.get().flatten(),  # Explicitly convert ALPHA to a numpy array
                    method='L-BFGS-B',
                    bounds = [(0, cp.max(P) * 1)] * ALPHA.size,  # Allow more flexibility in both directions
                    options={"disp": True, "maxiter": 100, "ftol": 1e-8, "gtol": 1e-5},
                )
                
                # Update ALPHA
                ALPHA = cp.array(result.x).reshape(ALPHA.shape)  # Convert result.x back to a cupy array

                
                # Check if we have a valid solution (non-trivial ALPHA values)
                if cp.sum(ALPHA > 0) > 0:
                    break
                
                print("Lambda is set too large (trivial solution)!\n\n")
                
                # Adjust LAMBDA
                LAMBDA = LAMBDA * 0.8
                print(f"Resetting to {LAMBDA}\n\n")
            
            print("Graph construction succeeded.\n\n")
            print("Number of non-zero edges:")
            print(cp.sum(ALPHA > 0))

            # Warnings for lambda settings
            ratio = cp.sum(ALPHA > 0) / cp.sum(P == True)
            if ratio > 0.9:
                print("Warning: Lambda may be set too small!")
            if ratio < 0.1:
                print("Warning: Lambda may be set too large!")
            
            # Symmetrize ALPHA
            ALPHA = ALPHA + ALPHA.T
            # Step 1: Create graph matrix G
            G = ALPHA / 1  # Equivalent to copying ALPHA
            cp.fill_diagonal(G, 0)  # Ensure diagonal is 0

            # Step 2: Compute the weight matrix W
            D = cp.diag(G.sum(axis=1))  # Convert D to a Cupy array
            W = cp.linalg.solve(cp.eye(G.shape[0]) + L_E * (D - G), cp.eye(G.shape[0]))
            
            if d:
                print("Weight matrix is forced to be diagonal\n")
                diag_W = np.diag(W).copy()
                W[G == 0] = 0
                np.fill_diagonal(W, diag_W)
                W /= W.sum(axis=1, keepdims=True)
            E = cp.array(E)  # Convert E to a Cupy array if it's a NumPy array
            print(E.shape)
            # Step 3: Denoise the expression matrix
            E_denoised = W @ E
            print(E_denoised.shape)
            # Step 4: Retrieve the embedding space
            print("Retrieving the embedding space...\n")
            # Create Q using Cupy
            Q = cp.eye(ALPHA.shape[0]) + 4 * cp.diag(G.sum(axis=1)) - 4 * G

            # Solve Q using Cupy
            Q_inv = cp.linalg.solve(Q, cp.eye(Q.shape[0]))

            # Symmetrize Q_inv
            Q_inv = (Q_inv + Q_inv.T) / 2

            # Create tmp with Cupy
            tmp = cp.ones_like(Q_inv) / Q_inv.shape[0]
            tmp = cp.eye(Q_inv.shape[0]) - tmp

            # Eigen decomposition
            decomposition_values, decomposition_vectors = eigh(tmp.get() @ Q_inv.get() @ tmp.get())
            Y = decomposition_vectors @ np.diag(decomposition_values**0.5)
            Y = Y[:, :K]  # Select the top K components
            print("Saving outputs...\n")
            # Extract row names from the input E matrix
            

            # Create DataFrame with row names for saving
            E_denoised_df = pd.DataFrame(E_denoised.get(), index=rownames_E)
            Y_df = pd.DataFrame(Y, index=rownames_E, columns=[f"Component{i+1}" for i in range(Y.shape[1])])
            G_df = pd.DataFrame(G.get(), index=rownames_E, columns=rownames_E)

            # File names
            E_denoised_fn = f"{project_name}_Denoised_matrix.txt"
            G_fn = f"{project_name}_Detected_graph.txt"
            Y_fn = f"{project_name}_Latent_space.txt"

            # Saving to files
            E_denoised_df.round(5).to_csv(os.path.join(output_path, E_denoised_fn), sep="\t", header=True, index=True)
            G_df.round(5).to_csv(os.path.join(output_path, G_fn), sep="\t", header=True, index=True)
            Y_df.round(5).to_csv(os.path.join(output_path, Y_fn), sep="\t", header=True, index=True)

            print("Outputs saved successfully.")
        """
        # Pack all parameters for denoise.R job
        sprod_cmd = ["Rscript", sprod_script]
        for sprod_key, param in zip(
            [
                "-e",
                "-c",
                "-f",
                "-p",
                "-n",
                "-r",
                "-u",
                "-k",
                "-l",
                "-t",
                "-m",
                "-x",
                "-d",
                "-s",
                "-o",
                "-p",
                "-g",
            ],
            [
                cts_fn,
                meta_fn,
                feature_fn,
                pnn_fn,
                sprod_npc,
                sprod_R,
                spord_perplexity,
                sprod_latent_dim,
                sprod_graph_reg,
                sprod_weight_reg,
                sprod_margin,
                sprod_umap,
                sprod_diag,
                sprod_path,
                os.path.abspath(output_path),
                patch_name,
                sprod_diagnose,
            ],
        ):
            if isinstance(param, bool):  # for toggles
                if param:
                    sprod_cmd.append(str(sprod_key))
            else:  # for kwargs
                sprod_cmd.append(sprod_key)
                sprod_cmd.append(str(param))
        # log file name is packed into cmd
        list_sprod_cmds.append(sprod_cmd + [log_fn])
    if len(list_sprod_cmds) == 1:
        sprod_worker(list_sprod_cmds[0])
    elif len(list_sprod_cmds) > 1:
        with Pool(16) as p:
            _ = p.map(sprod_worker, list_sprod_cmds)
        """


    # last step, stiching all subsamples back.
    if input_type == "batch":
        stiching_script_path = os.path.join(sprod_path, "slide_seq_stiching.py")
        stiching_subsampled_patches(
            output_path, os.path.join(output_path, "denoised_stitched.txt")
        )
