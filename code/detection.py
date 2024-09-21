import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import save_checkpoint
from fasterRCNN import fasterRCNN
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from detectionDataset import DetectionDataset, DetectionPrefetcher, collate_fn

# Constants
num_epochs = 2
num_classes = 1
batch_size = 4
img_width = 480 # 256
img_height = 270 # 144
learning_rate = 1e-4
momentum = 0.9
tresh = 0.38
overlap = 0.6
resume_training = True
show_examples = False
device = torch.device("cuda", 0)
save_folder = f'./results/detection/FastRCNN_{num_classes}_{img_width}x{img_height}/'
model_weights_path = save_folder + 'epoch_6_True.pth.tar'

def main():
    # Load data
    start = time.time()
    train_prefetcher = create_prefetcher('train')
    valid_prefetcher = create_prefetcher('valid')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    fastRcnn = fasterRCNN(num_classes)
    # fastRcnn.printModel()
    model = fastRcnn.model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1, verbose=False)

    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(train_prefetcher) - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    writer = SummaryWriter(save_folder)

    # Training
    best_acc = 0.0
    start_epoch = 0

    if resume_training:
        checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
  
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start = time.time()
        train(model, optimizer, train_prefetcher, epoch, writer, scheduler, lr_scheduler)
        map50 = validate(model, valid_prefetcher, epoch, writer)

        # Save model
        is_best = map50 > best_acc
        is_last = (epoch + 1) == (start_epoch + num_epochs)
        best_acc = max(map50, best_acc)
        if (is_best or is_last):
            save_checkpoint({"epoch": epoch + 1,
                            "best_acc1": best_acc,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()},
                            f"epoch_{epoch + 1}_{is_best}.pth.tar",
                            save_folder,
                            )
        end = time.time() - start
        print('Training epoch finished in (%.2fs)' % end)

def train(
    model, 
    optimizer, 
    train_prefetcher,
    epoch, 
    writer,
    scheduler,
    lr_scheduler
):
    batch_idx = 0
    dataset_size = len(train_prefetcher)

    model.train()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()        

    while batch_data is not None:
        batch_idx += 1
        images = list(image.to(device) for image in batch_data[0])
        target = [{k: v.to(device) for k, v in t.items()} for t in batch_data[1]]

        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, target)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        scheduler.step(epoch + (batch_idx/dataset_size))

        # Write the data to the log file
        if batch_idx % 20 == 0:
            writer.add_scalar("Train/Loss", loss_value, batch_idx + epoch * dataset_size)
            writer.add_scalar("Train/Loss_Classifier", loss_dict['loss_classifier'].detach().cpu(), batch_idx + epoch * dataset_size)
            writer.add_scalar("Train/Loss_Box_Reg", loss_dict['loss_box_reg'].detach().cpu(), batch_idx + epoch * dataset_size)
            writer.add_scalar("Train/Loss_Objectness", loss_dict['loss_objectness'].detach().cpu(), batch_idx + epoch * dataset_size)
            writer.add_scalar("Train/Loss_Rpn_Box_Reg", loss_dict['loss_rpn_box_reg'].detach().cpu(), batch_idx + epoch * dataset_size)
            print(f'Epoch: {epoch}, batch: {batch_idx}, loss = {loss_value}')

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

def validate(
    model, 
    valid_prefetcher,
    epoch,
    writer
):
    model.eval()
    valid_prefetcher.reset()
    batch_data = valid_prefetcher.next()
    metric = MeanAveragePrecision()

    batch_idx = 0
    targets = []
    preds = []
    with torch.no_grad():
        while batch_data is not None:
            batch_idx += 1
            images = list(image.to(device) for image in batch_data[0])
            target = [{k: v.to(device) for k, v in t.items()} for t in batch_data[1]]

            outputs = model(images)
            torch.cuda.synchronize()

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = target[i]['boxes'].detach().cpu()
                true_dict['labels'] = target[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                targets.append(true_dict)
            
            # Preload the next batch of data
            batch_data = valid_prefetcher.next()

    metric.update(preds, targets)
    metric_summary = metric.compute()
    print(f"Validation stats: map50={metric_summary['map_50']}, map75={metric_summary['map_75']}", )
    writer.add_scalar("Valid/MAP_50", metric_summary['map_50'], epoch + 1)
    writer.add_scalar("Valid/MAP_75", metric_summary['map_75'], epoch + 1)

    return metric_summary['map_50']

def create_prefetcher(mode) -> DetectionPrefetcher:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    isTrain = mode == 'train'

    dataset = DetectionDataset(mode, mean, std, num_classes, img_width, img_height)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=isTrain,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=isTrain,
                            collate_fn=collate_fn)
    return DetectionPrefetcher(dataloader, device)

def visualizeData():
    image = cv2.imread('data/Warp-D/train/images/Monitoring_photo1_04-Mar_03-09-16.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    
    boxes = []
    labels = []
    _, ax = plt.subplots()
    ax.imshow(image)

    with open('data/Warp-D/train/labels\Monitoring_photo1_04-Mar_03-09-16.txt') as f:
        for line in f:
            parsed = line.split()
            labels.append(int(parsed[0]))
            x = int(float(parsed[1])*1920)
            y = int(float(parsed[2])*1080)
            width = int(float(parsed[3])*1920)
            height = int(float(parsed[4])*1080)

            boxes.append([x-width/2, y-height/2, x+width/2, y+height/2])

    rect = patches.Rectangle((boxes[0][0], boxes[0][1]), boxes[0][2]-boxes[0][0], boxes[0][3]-boxes[0][1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((boxes[1][0], boxes[1][1]), boxes[1][2]-boxes[1][0], boxes[1][3]-boxes[1][1], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((boxes[2][0], boxes[2][1]), boxes[2][2]-boxes[2][0], boxes[2][3]-boxes[2][1], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    plt.show()

def test() -> None:
    # Load data
    start = time.time()    
    test_prefetcher = create_prefetcher('test')
    
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    fastRcnn = fasterRCNN(num_classes)
    model = fastRcnn.model.to(device)

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    metric = MeanAveragePrecision()

    batch_idx = 0
    targets = []
    preds = []
    start = time.time()
    with torch.no_grad():
        while batch_data is not None:
            batch_idx += 1
            images = list(image.to(device) for image in batch_data[0])
            target = [{k: v.to(device) for k, v in t.items()} for t in batch_data[1]]

            outputs = model(images)
            torch.cuda.synchronize()

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = target[i]['boxes'].detach().cpu()
                true_dict['labels'] = target[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                
                # Non-maximum supression
                idx = cv2.dnn.NMSBoxes(preds_dict['boxes'].tolist(), preds_dict['scores'].tolist(), tresh, overlap)

                preds_dict2 = dict()
                preds_dict2['boxes'] = preds_dict['boxes'][idx]
                preds_dict2['scores'] = preds_dict['scores'][idx]
                preds_dict2['labels'] = preds_dict['labels'][idx]                

                preds_dict2['labels'][preds_dict2['labels']>0] = 1
                true_dict['labels'][true_dict['labels']>0] = 1

                # preds.append(preds_dict)
                preds.append(preds_dict2)
                targets.append(true_dict)
                
                if show_examples and batch_idx % 10 == 0 and i % 4 == 0:
                    _, ax = plt.subplots()
                    img = images[i].numpy(force=True).T
                    img = np.swapaxes(img,0,1)
                    img -= img.min()
                    img /= img.max()
                    ax.imshow(img)

                    true_boxes = true_dict['boxes'].numpy(force=True)
                    pred_boxes = preds_dict2['boxes'].numpy(force=True)
                    true_labels = true_dict['labels'].numpy(force=True)
                    pred_labels = preds_dict2['labels'].numpy(force=True)
                    pred_scores = preds_dict2['scores'].numpy(force=True)

                    print(true_labels)
                    print(pred_labels)
                    print(pred_scores)
                    for j in range(true_boxes.shape[0]):
                        rect = patches.Rectangle((true_boxes[j][0], true_boxes[j][1]),
                                                 true_boxes[j][2]-true_boxes[j][0],
                                                 true_boxes[j][3]-true_boxes[j][1],
                                                 linewidth=1, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                    for j in range(pred_boxes.shape[0]):
                        rect = patches.Rectangle((pred_boxes[j][0], pred_boxes[j][1]), 
                                                 pred_boxes[j][2]-pred_boxes[j][0], 
                                                 pred_boxes[j][3]-pred_boxes[j][1],
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                    plt.show()
            
            # Preload the next batch of data
            batch_data = test_prefetcher.next()
    
    metric.update(preds, targets)
    metric_summary = metric.compute()
    print(f"Test stats: map50={metric_summary['map_50']}, map75={metric_summary['map_75']}", )
    end = time.time() - start
    print(f'Inference time per image {end/552}s')

    return metric_summary['map_50']

def valid_plots() -> None:
    # Load data
    start = time.time()    
    valid_prefetcher = create_prefetcher('valid')
    
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    fastRcnn = fasterRCNN(num_classes)
    model = fastRcnn.model.to(device)

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    valid_prefetcher.reset()
    batch_data = valid_prefetcher.next()

    batch_idx = 0
    targets = []
    preds = []
    treshs = []
    for lim in range(0,100,2):
        preds.append([])
        targets.append([])
        treshs.append(lim/100)
    
    with torch.no_grad():
        while batch_data is not None:
            batch_idx += 1
            images = list(image.to(device) for image in batch_data[0])
            target = [{k: v.to(device) for k, v in t.items()} for t in batch_data[1]]

            outputs = model(images)
            torch.cuda.synchronize()

            for i in range(len(images)):
                for j in range(len(treshs)):
                    true_dict = dict()
                    preds_dict = dict()
                    true_dict['boxes'] = target[i]['boxes'].detach().cpu()
                    true_dict['labels'] = target[i]['labels'].detach().cpu()
                    preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                    preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                    preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                    
                    idx = cv2.dnn.NMSBoxes(preds_dict['boxes'].tolist(), preds_dict['scores'].tolist(), tresh, treshs[j])

                    preds_dict2 = dict()
                    preds_dict2['boxes'] = preds_dict['boxes'][idx]
                    preds_dict2['scores'] = preds_dict['scores'][idx]
                    preds_dict2['labels'] = preds_dict['labels'][idx]                

                    preds_dict2['labels'][preds_dict2['labels']>0] = 1
                    true_dict['labels'][true_dict['labels']>0] = 1

                    # preds.append(preds_dict)
                    preds[j].append(preds_dict2)
                    targets[j].append(true_dict)
            
            # Preload the next batch of data
            batch_data = valid_prefetcher.next()
    
    map50 = []
    map75 = []
    mar = []
    for j in range(len(treshs)):
        metric = MeanAveragePrecision()
        metric.update(preds[j], targets[j])
        metric_summary = metric.compute()
        map50.append(metric_summary['map_50'])
        map75.append(metric_summary['map_75'])
        mar.append(metric_summary['mar_100'])

    plt.figure()
    plt.plot(treshs, map50)
    plt.plot(treshs, map75)
    plt.plot(treshs, mar)
    plt.title('Zavisnost MaP50 i MaP75 metrika od praga preklapanja detektora za prag od 0.38')
    plt.show()

    return

if __name__ == "__main__":
    visualizeData()
    main()
    # valid_plots()
    # test()

    # tensorboard --logdir=results/detection
    # python train.py --img 480 --batch 16 --epochs 30 --single-cls --data recycle.yaml --weights yolov5m.pt
    # python val.py --img 480 --batch 16 --single-cls --data recycle.yaml --weights runs\train\exp5\weights\best.pt --task test