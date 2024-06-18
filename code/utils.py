import os
import torch
import numpy as np
from torchvision.transforms import functional as F_vision

# Convert the image data type to the Tensor data type supported by PyTorch
def image_to_tensor(image: np.ndarray, mask: bool = False) -> torch.Tensor:
    if mask:
        image = np.array(image)
        image = image[:,:,0] > 0

    tensor = F_vision.to_tensor(image)

    return tensor

# Convert the Tensor data type supported by PyTorch to the image data type
def tensor_to_image(tensor: torch.Tensor, range_norm: bool) -> np.ndarray:
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


# Save Torch checkpoint
def save_checkpoint(
        state_dict: dict,
        file_name: str,
        dir_name: str,
) -> None:
    checkpoint_path = os.path.join(dir_name, file_name)
    torch.save(state_dict, checkpoint_path)
