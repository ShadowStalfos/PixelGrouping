import re
from io import BytesIO
from pathlib import Path
import random
from typing import TypedDict, List
import numpy as np
import torch
from PIL import Image
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from torchvision.transforms import transforms, ToTensor
import matplotlib.pyplot as plt
from read_bins import readColmapSceneInfo

DATA_DIR = Path("DataFor3DGS/")
OUTPUT_DIR = Path("datasets/DataFor3DGS/")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)
TRAIN_RATIO = 0.7

def split_directories(data_dir: Path, train_ratio: float = 0.7, seed: int = 42) -> (List[str], List[str]):
    random.seed(seed)
    all_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]  # Get only the names of directories
    random.shuffle(all_dirs)
    num_train = max(1, int(len(all_dirs) * train_ratio))
    train_dirs = all_dirs[:num_train]
    test_dirs = all_dirs[num_train:]
    return train_dirs, test_dirs

def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    if path.is_file():
        return path.stat().st_size
    total_size = 0
    for item in path.rglob('*'):  # Recursively go through all files and subdirectories
        if item.is_file():
            total_size += item.stat().st_size
    return total_size

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))

def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    return {filename_to_int(path.stem): load_raw(path) for path in example_path.iterdir()}

def filename_to_int(filename: str) -> int:
    # Use regular expression to find digits in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())  # Convert the first group of digits found to an integer
    raise ValueError(f"No numeric part found in filename: {filename}")

class Metadata(TypedDict):
    image_names: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]

class SceneRep(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def convert_pil_to_tensor(pil_img):
    """Convert PIL image to PyTorch tensor."""
    return transforms.ToTensor()(pil_img)


def show_image_and_mask(image_tensor, mask_tensor):
    """Plot an image and its corresponding mask."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image (assuming it is normalized to [0, 1] by ToTensor)
    ax[0].imshow(image_tensor.permute(1, 2, 0))  # Convert CxHxW to HxWxC
    ax[0].set_title('Image')
    ax[0].axis('off')

    # Check and plot the mask if it exists
    if mask_tensor is not None:
        # Assume the mask tensor is in CxHxW and single-channel
        mask_np = mask_tensor.squeeze().numpy()  # Reduce dimensions if needed
        ax[1].imshow(mask_np, cmap='gray')
        ax[1].set_title('Object Mask')
    else:
        ax[1].text(0.5, 0.5, 'No mask available', horizontalalignment='center', verticalalignment='center')

    ax[1].axis('off')
    plt.show()

class ImageProcessor:
    def __init__(self):
        self.to_tensor = ToTensor()  # Initialize the tensor transformation

    def convert_image(self, image):
        """Convert a list of raw byte images to a tensor of shape [batch, 3, height, width]."""
        pil_image = Image.open(BytesIO(image.numpy().tobytes()))  # Convert the raw bytes to a PIL Image
        tensor_image = self.to_tensor(pil_image)  # Convert the PIL Image to a PyTorch Tensor
        return tensor_image
    def convert_mask(self, mask):
        mask_pil = Image.open(BytesIO(mask.numpy().tobytes()))
        # Convert mask to a tensor: assumes that the mask is grayscale (1 channel)
        mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.int64)  # Use int64 for categorical data
        return mask_tensor


def chunk_stage(dirs, stage):
    chunk_size = 0
    chunk_index = 0
    chunk: list[SceneRep] = []

    def save_chunk(chunk, chunk_index, chunk_size, dirs, stage):
        chunk_key = f"{chunk_index:0>6}"
        print(
            f"Saving chunk {chunk_key} of {len(dirs)} ({chunk_size / 1e6:.2f} MB)."
        )
        dir = OUTPUT_DIR / stage
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(chunk, dir / f"{chunk_key}.torch")

        return [], chunk_index + 1, 0  # Return new empty chunk, incremented index, and reset size

    for scene_dir in dirs:
        image_dir = DATA_DIR / scene_dir / "images"
        object_dir = DATA_DIR / scene_dir / "object_mask"
        num_bytes = get_size(image_dir) + get_size(object_dir)

        # Read images and metadata.
        images = load_images(image_dir)
        objects = load_images(object_dir)

        cam_infos = readColmapSceneInfo(scene_path=scene_dir)
        scene_rep = {"image_names": None, "cameras": None}
        camera_data = []
        image_names = []

        for cam in cam_infos:
            # Extract the required data
            data = [
                cam.fx_norm, cam.fy_norm, cam.px_norm, cam.py_norm, 0.0, 0.0,
                cam.R[0, 0], cam.R[0, 1], cam.R[0, 2], cam.T[0],
                cam.R[1, 0], cam.R[1, 1], cam.R[1, 2], cam.T[1],
                cam.R[2, 0], cam.R[2, 1], cam.R[2, 2], cam.T[2]
            ]
            camera_data.append(data)
            image_names.append(filename_to_int(cam.image_name))

        camera_matrix = torch.tensor(np.stack(camera_data), dtype=torch.float32)
        image_names = torch.tensor(image_names, dtype=torch.int64)
        scene_rep["cameras"] = camera_matrix
        scene_rep["image_names"] = image_names
        scene_rep["images"] = [images[id.item()] for id in scene_rep["image_names"]]
        scene_rep["objects"] = [objects[id.item()] for id in scene_rep["image_names"]]

        # image_processor = ImageProcessor()
        #
        # first_image_tensor = image_processor.convert_image(scene_rep["images"][0])  # First image tensor
        # print(first_image_tensor)
        # print(first_image_tensor.shape)
        # first_mask_tensor = image_processor.convert_mask(scene_rep["objects"][0])  # First object mask tensor
        # print(first_mask_tensor)
        # print(first_mask_tensor.shape)
        # show_image_and_mask(first_image_tensor, first_mask_tensor)

        assert len(images) == len(scene_rep["image_names"])

        # Add the key to the example.
        scene_rep["key"] = scene_dir

        print(f"    Added {scene_dir} to chunk ({num_bytes / 1e6:.2f} MB).")
        chunk.append(scene_rep)
        chunk_size += num_bytes

        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            chunk, chunk_index, chunk_size = save_chunk(chunk, chunk_index, chunk_size, dirs, stage)

    if chunk_size > 0:
        chunk, chunk_index, chunk_size = save_chunk(chunk, chunk_index, chunk_size, dirs, stage)


if __name__ == "__main__":
    train_dirs, test_dirs = split_directories(DATA_DIR, TRAIN_RATIO)

    for stage, dirs in zip(("train", "test"), (train_dirs, test_dirs)):
        chunk_stage(dirs, stage)

