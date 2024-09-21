import time
import torch
import numpy as np
from denseNet import DenseNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from uNet import dice_numpy
from segmentationDataset import SegmentationDataset, SegmentationPrefetcher
from torchvision.transforms import functional as Ft, InterpolationMode

# Constants
batch_size = 28
netType = 121
learning_rate = 0.01
dropout_rate = 0.0 # 0.2
momentum = 0.9
weight_decay = 2e-05
label_smoothing = 0.1
T_mult = 1
eta_min = 5e-5
prepr = 0
weighted = False
gradCam = True
device = torch.device("cuda", 0)

resume_training = True
save_folder = f'./results/classification/DenseNet{netType}_{dropout_rate}_{learning_rate}_{prepr}_{weighted}/'
model_weights_path = save_folder + 'epoch_25_True.pth.tar'


def test():
    # Load data
    start = time.time()   
    test_prefetcher = create_prefetcher('test')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    model = DenseNet(netType=netType)
    model = model.to(device=device, memory_format=torch.channels_last)

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # Test
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    start = time.time()
    dice_score = 0

    while batch_data is not None:
        images = batch_data["image"].to(device=device, non_blocking=True)
        masks = batch_data["mask"].to(device=device, non_blocking=True)
        sizes = batch_data["max_size"].to(device=device, non_blocking=True)

        output = model(images) # batch_size x num_classes
        torch.cuda.synchronize()

        # Calculate accuracy
        pred = output.argmax(dim=1) # batch_size

        # Grad-cam
        if gradCam:
            output[:,pred[:]].sum().backward() # 64x64
            
            gradients = model.get_activations_gradient()
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # 1024
            activations = model.get_activations(images).detach() # 64x1024x7x7
            for i in range(1024):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze().cpu()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)
            
            for i in range(images.size(0)):
                bigsize = sizes[i]
                pomocni = heatmap[i]
                img = Ft.resize(images[i], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)
                msk = Ft.resize(masks[i], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)
                pred = Ft.resize(pomocni[None,:,:], (bigsize, bigsize), interpolation=InterpolationMode.BICUBIC)

                img = img.numpy(force=True).T
                msk = msk.numpy(force=True).T
                pred = pred.numpy(force=True).T
                pred = pred > 0.33*pred.max()
                pred = np.array(pred, np.float32)
                msk = np.array(msk, np.float32)

                dice_score += dice_numpy(pred[:,:,0], msk)

                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imshow(img)
                # plt.subplot(1, 3, 2)
                # plt.imshow(msk)
                # plt.subplot(1, 3, 3)
                # plt.imshow(pred)
                # plt.show()

        print(f"Test dice score is {dice_score/28*100}%")
        end = time.time() - start
        print(f'Inference time per image {end/images.size(0)/11}s')
                        
        # Preload the next batch of data
        batch_data = test_prefetcher.next()

def create_prefetcher(mode) -> SegmentationPrefetcher:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    isTrain = mode == 'train'

    dataset = SegmentationDataset(mode, 256, prepr, mean, std)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=isTrain,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=isTrain,
                            persistent_workers=True)
    return SegmentationPrefetcher(dataloader, device)

if __name__ == '__main__':
    test()