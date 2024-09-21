import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from utils import image_to_tensor
from torchvision import transforms
from dataset import ImageDataset, Prefetcher

class ClassificationDataset(ImageDataset):
    def __init__(self, mode: str, mean: list, std: list, prepr: int=0, gradCam: bool=False) -> None:
        super().__init__(mode, mean, std)
        self.image_file_paths = glob(f"data/Warp-C/{mode}_crops/*/*/*")
        classes = []
        with open('data/Warp-D/classes.txt') as f:
            for line in f:
                classes.append(line[:-1])
        self.class_to_idx = dict([(y,x) for x,y in enumerate(sorted(set(classes)))])

        self.image_size = 224
        self.mode = mode
        self.prepr = prepr
        self.gradCam = gradCam and mode == 'test'

        if self.mode == "train":
            self.pre_transform = transforms.Compose([
                transforms.Resize(200, max_size=self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        else:
            self.pre_transform = transforms.Compose([
                transforms.Resize(200, max_size=self.image_size),
                transforms.CenterCrop(self.image_size),
            ])

    def get_weights(self) -> list:
        num_classes = len(self.class_to_idx)

        numSamples = np.zeros((num_classes,))
        for img in self.image_file_paths:
            img_dir, _ = img.split('\\')[-2:]
            numSamples[self.class_to_idx[img_dir]] += 1

        return np.sum(numSamples) / (numSamples[:] * 28)

    def __getitem__(self, batch_index: int): #  -> Union[torch.Tensor, torch.Tensor, int]
        image_dir, _ = self.image_file_paths[batch_index].split('\\')[-2:]
        image = cv2.imread(self.image_file_paths[batch_index])
        if self.prepr & 1:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.class_to_idx[image_dir]

        image = Image.fromarray(image)
        image = self.pre_transform(image)
        img_tensor = image_to_tensor(image)
        tensor = self.post_transform(img_tensor)

        if self.gradCam:
            return {"image": tensor, "og_image": img_tensor, "target": target}
        return {"image": tensor, "target": target}

    
class ClassificationPrefetcher(Prefetcher):
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
