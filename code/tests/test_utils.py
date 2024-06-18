import os
import torch
import numpy as np
from code_v2.utils import image_to_tensor, tensor_to_image, save_checkpoint

def test_image_to_tensor() -> None:
    """
    Test that image to tensor returns torch tensor.
    """
    mock_img = np.ones((5,4,3))
    output = image_to_tensor(mock_img)
    assert isinstance(output, torch.Tensor)

def test_image_to_tensor_mask() -> None:
    """
    Test that image to tensor returns binary mask.
    """
    mock_img = np.ones((5,4,3))
    output = image_to_tensor(mock_img, True)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 5, 4)

def test_tensor_to_image() -> None:
    """
    Test that tensor to image returns permuted unit8 matrix.
    """
    mock_img = np.ones((3,4,5))
    mock_tensor = torch.Tensor(mock_img)
    output = tensor_to_image(mock_tensor, False)
    assert isinstance(output, np.ndarray)
    assert output.shape == (4, 5, 3)
    assert isinstance(output[0,0,0], np.uint8)
    assert output[0,0,0] == 255

def test_tensor_to_image_norm() -> None:
    """
    Test that tensor to image returns permuted unit8 matrix.
    """
    mock_img = -np.ones((3,4,5))
    mock_tensor = torch.Tensor(mock_img)
    output = tensor_to_image(mock_tensor, True)
    assert isinstance(output, np.ndarray)
    assert output.shape == (4, 5, 3)
    assert isinstance(output[0,0,0], np.uint8)
    assert output[0,0,0] == 0

def test_save_checkpoint() -> None:
    """
    Test that save checkpoint writes a checkpoint file.
    """
    mock_dict = {}
    mock_name = 'test_file.pth.tar'
    save_checkpoint(mock_dict, mock_name,'')
    assert os.path.exists(mock_name)