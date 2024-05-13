import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch
import torchvision.transforms.functional as F
from ..types import AnyExample, AnyViews


def rescale(image: Float[Tensor, "3 h_in w_in"], shape: tuple[int, int], method='LANCZOS') -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    if method == 'NEAREST':
        image_new = image_new.resize((w, h), Image.NEAREST)
    else:
        image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics

def rescale_and_crop(images: Float[Tensor, "*#batch c h w"], intrinsics: Float[Tensor, "*#batch 3 3"], shape: tuple[int, int], imgs_nearest_neighbors=False) -> tuple:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    rescale_method = 'NEAREST' if imgs_nearest_neighbors else 'LANCZOS'
    images = torch.stack([rescale(image, (h_scaled, w_scaled), method=rescale_method) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def center_crop_masks(
    masks: torch.Tensor,  # Expected to be an integer tensor
    shape: tuple[int, int],
) -> torch.Tensor:
    *_, h_in, w_in = masks.shape
    h_out, w_out = shape
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Perform the cropping operation
    cropped_masks = masks[..., :, row : row + h_out, col : col + w_out]

    return cropped_masks

def rescale_masks(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Rescale a batch of masks using nearest neighbor interpolation.

    Args:
        masks (torch.Tensor): The batch of masks tensors to rescale.
        shape (tuple[int, int]): The target shape (height, width).

    Returns:
        torch.Tensor: The resized batch of mask tensors.
    """
    # Check if masks already include a batch dimension and adjust if not
    if masks.dim() == 2:  # No batch, single mask without channel
        masks = masks.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
    elif masks.dim() == 3:  # No batch, but masks may already have a channel dimension
        masks = masks.unsqueeze(1)  # Add batch dimension

    # Resize using nearest neighbor interpolation to preserve label integrity
    resized_masks = F.resize(masks, list(shape), interpolation=F.InterpolationMode.NEAREST)

    return resized_masks

def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], imgs_nearest_neighbors=False) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape, imgs_nearest_neighbors=imgs_nearest_neighbors)
    updated_views = {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }
    if "objects" in views:
        # Rescale the masks to the desired shape using nearest neighbor interpolation
        masks = rescale_masks(views["objects"], shape)

        # Perform a center crop on the resized masks
        masks = center_crop_masks(masks, shape)

        # Update the views dictionary with the processed masks
        updated_views["objects"] = masks
    return updated_views



def apply_crop_shim(example: AnyExample, shape: tuple[int, int], imgs_nearest_neighbors=False) -> AnyExample:
    """Crop images and masks in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape, imgs_nearest_neighbors=imgs_nearest_neighbors),
        "target": apply_crop_shim_to_views(example["target"], shape, imgs_nearest_neighbors=imgs_nearest_neighbors),
    }

