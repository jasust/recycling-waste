from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class fasterRCNN():
    def __init__(self, num_classes):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, num_classes+1)

    def printModel(self):
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Ukupno {total_params} parametara.")
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Od toga {total_trainable_params} trenirajucih parametara.")
