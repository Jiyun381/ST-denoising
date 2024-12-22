import os
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from anndata import AnnData
import psutil

def tiling_optimized(
    adata: AnnData,
    out_path: Union[Path, str] = "./tiling",
    library_id: Union[str, None] = None,
    crop_size: int = 40,
    target_size: int = 299,
    img_fmt: str = "JPEG",
    verbose: bool = False,
    copy: bool = False,
    num_workers: int = 40,
    batch_size: int = 1200  # Adjust batch size as needed
) -> Optional[AnnData]:
    """\
    Optimized tiling of H&E images into smaller tiles based on spot spatial locations,
    with batch processing to reduce memory usage.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    out_path : Union[Path, str], optional
        Path to save spot image tiles, by default "./tiling".
    library_id : Union[str, None], optional
        Library id stored in AnnData, by default None.
    crop_size : int, optional
        Size of tiles to crop, by default 40.
    target_size : int, optional
        Desired size for the convolutional neural network input, by default 299.
    img_fmt : str, optional
        Image format to save tiles ('JPEG' or 'PNG'), by default "JPEG".
    verbose : bool, optional
        If True, prints detailed information, by default False.
    copy : bool, optional
        If True, returns a copy of adata with tile paths, by default False.
    num_workers : int, optional
        Number of parallel threads for processing, by default 40.
    batch_size : int, optional
        Number of tiles processed per batch, by default 1000.

    Returns
    -------
    Optional[AnnData]
        Updated AnnData object with tile paths if copy is True, else None.
    """
    def print_memory_usage(stage=""):
        mem = psutil.virtual_memory()
        print(f"Memory Usage at {stage}:")
        print(f"  Total: {mem.total / (1024 ** 3):.2f} GB")
        print(f"  Available: {mem.available / (1024 ** 3):.2f} GB")
        print(f"  Used: {mem.used / (1024 ** 3):.2f} GB")
        print(f"  Percentage Used: {mem.percent}%\n")

    print_memory_usage("Before Processing")

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Ensure the output directory exists
    Path(out_path).mkdir(parents=True, exist_ok=True)

    print_memory_usage("Before Loading Image")
    
    # Load and preprocess the image
    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]
    if image.dtype in [np.float32, np.float64]:
        image = (image * 255).astype(np.uint8)

    print_memory_usage("After Loading and Converting Image")
    
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")
        
    print_memory_usage("After Converting to PIL Image")
    
    # Extract spot coordinates
    spots = list(zip(adata.obs["imagerow"], adata.obs["imagecol"]))

    def process_tile(imagerow: float, imagecol: float) -> Optional[str]:
        """Processes a single tile: crops, resizes, and saves the image."""
        try:
            # Calculate bounding box
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2

            # Crop the image
            tile = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))

            # Resize the tile
            tile = tile.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Generate tile name with two decimal places for coordinates
            tile_name = f"{imagecol:.2f}-{imagerow:.2f}-{crop_size}"

            # Define output path
            tile_filename = f"{tile_name}.{img_fmt.lower()}"
            out_tile = Path(out_path) / tile_filename

            # Save the tile with optimized parameters
            if img_fmt.upper() == "JPEG":
                tile.save(out_tile, "JPEG", quality=85, optimize=True)
            else:
                tile.save(out_tile, img_fmt.upper(), optimize=True)

            if verbose:
                print(f"Generated tile at location ({imagecol}, {imagerow})")

            return str(out_tile)
        except Exception as e:
            if verbose:
                print(f"Error processing tile at ({imagecol}, {imagerow}): {e}")
            return None

    def chunker(seq, size):
        """Yield successive chunks of size `size` from seq."""
        for pos in range(0, len(seq), size):
            yield seq[pos:pos+size]

    tile_paths = []
    total_spots = len(spots)
    batch_count = 0

    # Process in batches
    for batch in chunker(spots, batch_size):
        batch_count += 1
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_tile, row, col): (row, col) for row, col in batch}

            # Use tqdm for progress within the batch
            desc_text = f"Tiling image batch {batch_count}"
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc_text, bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
                try:
                    tile_path = future.result()
                    tile_paths.append(tile_path)
                except Exception as e:
                    row, col = futures[future]
                    if verbose:
                        print(f"Error processing tile at ({col}, {row}): {e}")

    # Assign tile paths to adata.obs
    adata.obs["tile_path"] = tile_paths
    return adata.copy() if copy else None