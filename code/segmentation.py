import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import save_checkpoint
from torch.utils.data import DataLoader
from uNet import UNet, dice_coeff, dice_numpy
from skimage.restoration import denoise_tv_bregman
from torch.utils.tensorboard import SummaryWriter
from segmentationDataset import SegmentationDataset, SegmentationPrefetcher
from torchvision.transforms import functional as Ft, InterpolationMode

num_epochs = 40
batch_size = 28
learning_rate = 1e-5
momentum = 0.9
weight_decay = 1e-06
base_size = 64
bilinear = True
imgsize = 256
preprocessing = 1
device = torch.device("cuda", 0)

resume_training = False
save_folder = f'./results/segmentation/UNet_{base_size}_{bilinear}_{imgsize}_{preprocessing}/'
model_weights_path = save_folder + 'epoch_39_True.pth.tar'

def main():
    # Load data
    start = time.time()    
    train_prefetcher = create_prefetcher('train')
    valid_prefetcher = create_prefetcher('valid')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    model = UNet(base=base_size, bilinear=bilinear)
    model = model.to(device=device, memory_format=torch.channels_last)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(save_folder)

    # Training
    best_dice = 0.0
    start_epoch = 0

    if resume_training:
        state_dict = torch.load(model_weights_path, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        start = time.time()
        train(model, train_prefetcher, criterion, optimizer, epoch, scaler, writer)
        dice = validate(model, valid_prefetcher, epoch, writer)

        # Update LR
        scheduler.step(dice)

        # Save model
        is_best = dice > best_dice
        is_last = (epoch + 1) == (start_epoch + num_epochs)
        best_dice = max(dice, best_dice)
        if (is_best or is_last):
            save_checkpoint({"epoch": epoch + 1,
                            "best_dice": best_dice,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()},
                            f"epoch_{epoch + 1}_{is_best}.pth.tar",
                            save_folder,
                            )
        end = time.time() - start
        print('Training epoch finished in (%.2fs)' % end)

# Train function
def train(
        model: torch.nn.Module,
        train_prefetcher: SegmentationPrefetcher,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epoch: int,
        scaler: torch.cuda.amp.GradScaler,
        writer: SummaryWriter
) -> None:
    batch_idx = 0

    # Load the first batch of data
    model.train()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        images = batch_data["image"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
        mask = batch_data["mask"].to(device=device, non_blocking=True)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Training
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.squeeze(1), mask.float().squeeze(1))
            dice = dice_coeff(F.sigmoid(output.squeeze(1)), mask.float().squeeze(1))
            # loss += 1 - dice

        # Backpropagation
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Write the data to the log file
        batch_idx += 1
        writer.add_scalar("Train/Loss", loss.item(), batch_idx + epoch * 12)
        writer.add_scalar("Train/Dice", dice.item(), batch_idx + epoch * 12)
        print(f'Epoch: {epoch}, batch: {batch_idx}, loss = {loss.item()}, dice = {dice.item()}%')

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

def validate(model: torch.nn.Module,
        valid_prefetcher: SegmentationPrefetcher,
        epoch: int,
        writer: SummaryWriter
) -> float:
    model.eval().eval()
    valid_prefetcher.reset()
    batch_data = valid_prefetcher.next()
    dice_score = 0

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device=device, non_blocking=True)
            masks = batch_data["mask"].to(device=device, non_blocking=True)

            mask_pred = model(images)
            torch.cuda.synchronize()

            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            dice_score += dice_coeff(mask_pred.squeeze(1), masks.float().squeeze(1))

            batch_data = valid_prefetcher.next()

    writer.add_scalar("Valid/Dice", dice_score, epoch + 1)
    print(f"Validation dice coefficient after {epoch + 1} epochs is {dice_score}")

    return dice_score

def test():
    # Load data
    start = time.time()    
    test_prefetcher = create_prefetcher('test')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    model = UNet(base=base_size, bilinear=bilinear)
    model = model.to(device=device, memory_format=torch.channels_last)

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # Test
    dice_score = np.zeros((11, 1))
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    start = time.time()

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device=device, non_blocking=True)
            masks = batch_data["mask"].to(device=device, non_blocking=True)
            sizes = batch_data["max_size"].to(device=device, non_blocking=True)

            mask_pred = model(images)
            torch.cuda.synchronize()
            for j in range(11):
                mask_pred1 = (F.sigmoid(mask_pred) > 0.1*j).float()

                for i in range(images.size(0)):
                    bigsize = sizes[i]
                    img = Ft.resize(images[i], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)
                    msk = Ft.resize(masks[i], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)
                    pred = Ft.resize(mask_pred1[i], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)

                    img = img.numpy(force=True).T
                    msk = msk.numpy(force=True).T
                    pred = pred.numpy(force=True).T

                    # Post processing
                    kernel = np.ones((7, 7), np.float32)
                    msk = np.array(msk, np.float32)

                    pred = cv2.erode(np.array(pred > 0.1*j, np.float32), kernel)
                    pred = cv2.dilate(np.array(pred, np.float32), kernel)
                    pred = np.array(pred, np.float32)

                    dice_score[j] += dice_numpy(pred, msk)

                    if j == 3:
                        plt.figure()
                        plt.subplot(1, 3, 1)
                        plt.imshow(img)
                        plt.subplot(1, 3, 2)
                        plt.imshow(msk)
                        plt.subplot(1, 3, 3)
                        plt.imshow(pred)
                        plt.suptitle('Plava flasa')
                        plt.show()

            print(f"Test dice score is {dice_score.max()/28*100}%")
            end = time.time() - start
            print(f'Inference time per image {end/images.size(0)/11}s')

            plt.figure()
            plt.plot(np.arange(11)*0.1, dice_score/28)
            plt.title('Dice skor u zavisnosti od praga segmentacije')
            plt.ylabel('Dice skor')
            plt.xlabel('Prag segmentacije')
            plt.show()
                    
            # Preload the next batch of data
            batch_data = test_prefetcher.next()

def create_prefetcher(mode) -> SegmentationPrefetcher:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    isTrain = mode == 'train'

    dataset = SegmentationDataset(mode, imgsize, preprocessing, mean, std)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=isTrain,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=isTrain,
                            persistent_workers=True)
    return SegmentationPrefetcher(dataloader, device)

def visualizeData():
    image = cv2.imread('data/Warp-S/train_images/bottle-blue_test_Monitoring_photo_2_test_25-Mar_11-48-33_01.jpg')
    mask = cv2.imread('data/Warp-S/train_masks/bottle-blue_test_Monitoring_photo_2_test_25-Mar_11-48-33_01.png')
    image_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_corrected = mask[:,:,2]

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image_equ = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    image_breg = denoise_tv_bregman(image_corrected/255.0, 4)
    image_bil = cv2.bilateralFilter(image_corrected, 10, 50, 50)
    image_all = cv2.bilateralFilter(image_equ, 15, 75, 75)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(image_corrected)
    plt.subplot(2, 3, 2)
    plt.imshow(mask_corrected)
    plt.subplot(2, 3, 3)
    plt.imshow(image_equ)
    plt.subplot(2, 3, 4)
    plt.imshow(image_breg)
    plt.subplot(2, 3, 5)
    plt.imshow(image_bil)
    plt.subplot(2, 3, 6)
    plt.imshow(image_all)
    plt.suptitle('Plava flasa')
    plt.show()

if __name__ == '__main__':
    # visualizeData()
    # main()
    test()
    
    # tensorboard --logdir=results/segmentation
