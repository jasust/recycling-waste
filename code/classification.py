import cv2
import time
import torch
import numpy as np
from denseNet import DenseNet
import matplotlib.pyplot as plt
from utils import save_checkpoint
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from classificationDataset import ClassificationDataset, ClassificationPrefetcher

# Constants
num_epochs = 4
batch_size = 64
netType = 121
learning_rate = 0.01
dropout_rate = 0.0 # 0.2
momentum = 0.9
weight_decay = 2e-05
label_smoothing = 0.1
T_mult = 1
eta_min = 5e-5
prepr = 1
weighted = False
device = torch.device("cuda", 0)

resume_training = False
save_folder = f'./results/classification/DenseNet{netType}_{dropout_rate}_{learning_rate}_{prepr}_{weighted}/'
model_weights_path = save_folder + 'epoch_24_True.pth.tar'

def main():
    # Load data
    start = time.time()   
    train_prefetcher, weights = create_prefetcher('train')
    valid_prefetcher, _ = create_prefetcher('valid')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    model = DenseNet(netType=netType, dropout_rate=dropout_rate)
    model = model.to(device=device, memory_format=torch.channels_last)

    if weighted:
        weights = torch.tensor(weights, dtype=torch.float)
    else:
        weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    criterion = criterion.to(device=device, memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult, eta_min)

    writer = SummaryWriter(save_folder)

    scaler = torch.cuda.amp.GradScaler()

    # Training
    best_acc = 0.0
    start_epoch = 0

    if resume_training:
        checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc1"]
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    for epoch in range(start_epoch, start_epoch+num_epochs):
        start = time.time()
        train(model, train_prefetcher, criterion, optimizer, epoch, scaler, writer)
        acc = validate(model, valid_prefetcher, epoch, writer)

        # Update LR
        scheduler.step()

        # Save model
        is_best = acc > best_acc
        is_last = (epoch + 1) == (start_epoch + num_epochs)
        best_acc = max(acc, best_acc)
        if (is_best or is_last):
            save_checkpoint({"epoch": epoch + 1,
                            "best_acc1": best_acc,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()},
                            f"epoch_{epoch + 1}_{is_best}.pth.tar",
                            save_folder,
                            )
        end = time.time() - start
        print('Training epoch finished in (%.2fs)' % end)

# Vizualize data
def visualizeData():
    image = cv2.imread('data/Warp-C/train_crops/bottle/bottle-blue/POSAD_1_11-Sep_08-46-51_01.jpg')
    image_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_corrected)
    plt.suptitle('Plava flasa')
    plt.show()

    image = cv2.imread('data/Warp-C/train_crops/canister/canister/Robo_25-Mar_15-00-13_01.jpg')
    image_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_corrected)
    plt.suptitle('Kanister')
    plt.show()

# Train function
def train(
        model: torch.nn.Module,
        train_prefetcher: ClassificationPrefetcher,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epoch: int,
        scaler: torch.cuda.amp.GradScaler,
        writer: SummaryWriter
) -> None:
    batch_idx = 0
    cum_acc = 0
    num_examples = 0
    dataset_size = len(train_prefetcher)

    # Load the first batch of data
    model.train()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        images = batch_data["image"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=device, non_blocking=True)
        batch_size1 = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Training
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        _, pred = output.topk(1, dim=1)
        acc = pred.t().eq(target).sum()
        cum_acc += acc
        num_examples += batch_size1

        # Write the data to the log file
        batch_idx += 1
        if batch_idx % 10 == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_idx + epoch * dataset_size)
            writer.add_scalar("Train/Accuracy", cum_acc*100/num_examples, batch_idx + epoch * dataset_size)
            print(f'Epoch: {epoch}, batch: {batch_idx}, loss = {loss.item()}, accuracy = {cum_acc*100/num_examples}%')

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

def validate(
        model: torch.nn.Module,
        data_prefetcher: ClassificationPrefetcher,
        epoch: int,
        writer: SummaryWriter
) -> float:
    cum_acc = 0
    num_examples = 0
    
    # Load the first batch of data
    model.eval()
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device=device, non_blocking=True)
            target = batch_data["target"].to(device=device, non_blocking=True)

            batch_size1 = images.size(0)
            output = model(images)
            torch.cuda.synchronize()

            # Calculate accuracy
            _, pred = output.topk(1, dim=1)
            acc = pred.t().eq(target).sum()
            cum_acc += acc
            num_examples += batch_size1

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

    writer.add_scalar("Valid/Accuracy", cum_acc*100/num_examples, epoch + 1)
    print(f"Validation accuracy after {epoch + 1} epochs is {cum_acc*100/num_examples}%")

    return cum_acc*100/num_examples

def test():
    # Load data
    start = time.time()   
    test_prefetcher, _ = create_prefetcher('test')
    end = time.time() - start
    print('Loading data finished... (%.2fs)' % end)

    # Define model
    model = DenseNet(netType=netType)
    model = model.to(device=device, memory_format=torch.channels_last)

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # Test
    cum_acc = 0
    num_examples = 0
    dataset_size = 1551
    targets = np.zeros((dataset_size,1))
    predictions = np.zeros((dataset_size,1))

    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    start = time.time()

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device=device, non_blocking=True)
            target = batch_data["target"].to(device=device, non_blocking=True)

            batch_size1 = images.size(0)
            output = model(images)
            torch.cuda.synchronize()

            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = pred.t().eq(target).sum()
            targets[num_examples:num_examples+batch_size1] = np.reshape(target.numpy(force=True),(batch_size1,1))
            predictions[num_examples:num_examples+batch_size1] = np.reshape(pred.t().numpy(force=True),(batch_size1,1))
            cum_acc += acc
            num_examples += batch_size1

            # Grad-cam
            # pred.backward()
            # gradients = model.get_activations_gradient()
            # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            # activations = model.get_activations(images).detach()
            # print(activations)
            # for i in range(512):
            #     activations[:, i, :, :] *= pooled_gradients[i]
            # heatmap = torch.mean(activations, dim=1).squeeze()
            # heatmap = np.maximum(heatmap, 0)
            # heatmap /= torch.max(heatmap)
            
            # plt.figure
            # plt.matshow(heatmap.squeeze())
            # plt.show()

            # Preload the next batch of data
            batch_data = test_prefetcher.next()

    print(f"Test accuracy is {cum_acc*100/num_examples}%")
    end = time.time() - start
    print(f'Inference time per image {end/dataset_size}s')

    f1scores = f1_score(targets, predictions, average=None)
    print('F1 scores per class: ', f1scores)

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true=targets, y_pred=predictions)
    _, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center')
    
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)
    plt.show()

def create_prefetcher(mode) -> ClassificationPrefetcher:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    isTrain = mode == 'train'

    dataset = ClassificationDataset(mode, mean, std, prepr)
    weights = dataset.get_weights() if isTrain else []

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=isTrain,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=isTrain,
                            persistent_workers=True)
    return ClassificationPrefetcher(dataloader, device), weights

if __name__ == "__main__":
    # visualizeData()
    main()
    # test()
    
    # tensorboard --logdir=results/classification/new


