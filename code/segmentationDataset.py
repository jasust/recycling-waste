import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from utils import image_to_tensor
from torchvision import transforms
from skimage.restoration import denoise_tv_bregman
from dataset import ImageDataset, Prefetcher

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(Image.open(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)


class SegmentationDataset(ImageDataset):
    def __init__(self, mode: str, size: int, prepr: int, mean: list, std: list) -> None:
        super().__init__(mode, mean, std)
        if mode=='train':
            self.image_file_paths = glob(f"data/Warp-S/{mode}_images_{size}/*")
            self.mask_file_paths = glob(f"data/Warp-S/{mode}_masks_{size}/*")
        else:
            self.image_file_paths = glob(f"data/Warp-S/valid_images/*")
            self.mask_file_paths = glob(f"data/Warp-S/valid_masks/*") 
        self.pre_transform = transforms.Compose([
                transforms.Resize(250, max_size=size),
                transforms.CenterCrop(size),
            ])
        self.image_size = size
        self.prepr = prepr
        self.mode = mode

    def __getitem__(self, idx: int) -> [torch.Tensor, torch.Tensor, int]:
        image = cv2.imread(self.image_file_paths[idx])
        mask = cv2.imread(self.mask_file_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        max_size = max(image.shape)

        # Preprocessing - histogram equalization and denoising
        if self.prepr & 1:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.prepr & 2:
            denoised = denoise_tv_bregman(image/255.0, 4)
            image = (denoised*255).astype(np.uint8)
        if self.prepr & 4:
            image = cv2.bilateralFilter(image, 15, 75, 75)       

        image = Image.fromarray(image)
        if self.mode != 'train':
            image = self.pre_transform(image)
        image_tensor = image_to_tensor(image)
        image_tensor = self.post_transform(image_tensor)

        mask = Image.fromarray(mask)
        mask = self.pre_transform(mask)
        mask_tensor = image_to_tensor(mask, True)

        return {'image': image_tensor, 'mask': mask_tensor, 'max_size': max_size}
    
class SegmentationPrefetcher(Prefetcher):
    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)
