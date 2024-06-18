import cv2
import torch
from glob import glob
from utils import image_to_tensor
from dataset import ImageDataset, Prefetcher

class DetectionDataset(ImageDataset):
    def __init__(self, mode: str, mean: list, std: list, num_classes: int, width=256, height=144) -> None:
        super().__init__(mode, mean, std)
        self.image_file_paths = glob(f"data/Warp-D/{mode}/images/resized/*")
        self.annot_file_paths = glob(f"data/Warp-D/{mode}/labels/*")
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def load_image_and_labels(self, index):
        image_path = self.image_file_paths[index]
        annot_path = self.annot_file_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.width, self.height))
        
        boxes = []
        labels = []
        with open(annot_path) as f:
            for line in f:
                parsed = line.split()
                if len(parsed)>4:
                    labels.append(self.classToLabel(int(parsed[0])))
                    x = int(float(parsed[1])*self.width)
                    y = int(float(parsed[2])*self.height)
                    width = int(float(parsed[3])*self.width)
                    height = int(float(parsed[4])*self.height)

                    boxes.append([x-width/2, y-height/2, x+width/2, y+height/2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, boxes, labels, area

    def classToLabel(self, cls: int) -> int:
        if self.num_classes == 1:
            return 1
        if self.num_classes == 28:
            return cls+1
        if cls == 8:
            return 3
        if cls == 14:
            return 2
        if cls == 9 or cls == 10:
            return 4
        if cls == 11 or cls == 12 or cls == 13 or cls == 22:
            return 5
        return 1

    def __getitem__(self, idx):
        image, boxes, labels, area, = self.load_image_and_labels(idx)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        tensor = image_to_tensor(image)
        tensor = self.post_transform(tensor) 

        return tensor, target

class DetectionPrefetcher(Prefetcher):
    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for i in range(len(self.batch_data[0])):
                if torch.is_tensor(self.batch_data[0][i]):
                    self.batch_data[0][i] = self.batch_data[0][i].to(self.device, non_blocking=True)
            for j in range(len(self.batch_data[1])):
                for k, v in self.batch_data[1][j].items():
                    if torch.is_tensor(v):
                        self.batch_data[1][j][k] = self.batch_data[1][j][k].to(self.device, non_blocking=True)
    
def collate_fn(batch):
    return tuple(zip(*batch))
