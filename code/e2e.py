import cv2
import time
import torch
import matplotlib
import numpy as np
from PIL import Image
from denseNet import DenseNet
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.patches as patches

# Constants
width = 480
height = 270
scale = 2
numClasses = 28
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
cmaplist = [plt.cm.jet(i/numClasses) for i in range(numClasses)]
classes = []
with open('data/Warp-D/classes.txt') as f:
    for line in f:
        classes.append(line[:-1])
clsmap = [0,23,24,15,2,17,1,18,3,20,5,19,7,21,4,16,6,14,8,13,11,12,22,26,27,25,9,10]

# Parameters
showResults = False
fileName = 'POSAD_1_12-Sep_09-24-59'
fullImage = f'data/Warp-D/test/images/{fileName}.jpg'
imageName = f'yolov5/datasets/test/images/{fileName}.jpg'
label = f'yolov5/datasets/test/labels/{fileName}.txt'
model_weights_path = './results/classification/DenseNet121_0.0_0.01_0_False/epoch_25_True.pth.tar'

pre_transform = transforms.Compose([transforms.Resize(200, max_size=224), transforms.CenterCrop(224)])
post_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean, std)])

def visualizeBoxes(image, boxes, title=''):
    _, ax = plt.subplots()
    plt.suptitle(title)
    
    ax.imshow(image)
    for box in boxes:
        if len(box) > 5 and box[4] < 0.4:
            continue
        color = cmaplist[int(box[4])]
        rect = patches.Rectangle((scale*(box[0]-box[2]/2), scale*(box[1]-box[3]/2)), scale*box[2], scale*box[3], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        classText = classes[box[4]] if isinstance(box[4], int) else 'item'
        ax.text(scale*(box[0]+box[2]/2)-8, scale*(box[1]-box[3]/2)-8, classText, fontsize=11, color='w', 
                horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor=color, edgecolor=color))
    plt.show()

def getCroppedImage(image, box) -> torch.Tensor:
    cropped = image.crop((scale*(box[0]-box[2]/2), scale*(box[1]-box[3]/2), scale*(box[0]+box[2]/2), scale*(box[1]+box[3]/2)))
    cropped = pre_transform(cropped)
    img_tensor = transforms.functional.to_tensor(cropped)
    img_tensor = post_transform(img_tensor)
    return img_tensor

def centerCropImage(img, w, h):
    bigSize = int(max(w, h))
    result = cv2.resize(img, (bigSize, bigSize))
    x = max(bigSize/2 - w/2,0)
    y = max(bigSize/2 - h/2,0)
    return result[int(y):int(y+h), int(x):int(x+w)]

boxes = []
with open(label) as f:
    for line in f:
        parsed = line.split()
        if not len(parsed):
            continue
        l = int(parsed[0])
        x = int(float(parsed[1])*width)
        y = int(float(parsed[2])*height)
        w = int(float(parsed[3])*width)
        h = int(float(parsed[4])*height)
        boxes.append([x, y, w, h, l])

# Visualize label data
imageDetection = cv2.imread(fullImage)
imageDetection = cv2.cvtColor(imageDetection, cv2.COLOR_BGR2RGB)
visualizeBoxes(imageDetection, boxes, 'Labelirana fotografija')

# Load models
detModel = torch.hub.load('yolov5/yolov5/', 'custom', path='yolov5/yolov5/runs/train/exp2/weights/best.pt', source='local')
classModel = DenseNet(netType=121)
classModel = classModel.to(device=torch.device("cuda", 0), memory_format=torch.channels_last)
checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
classModel.load_state_dict(checkpoint["state_dict"])
classModel.eval()
matplotlib.use('QtAgg')

# Measure performance
start = time.time()

# Detection
results = detModel(imageName)
detectedBoxes = results.xywh[0].cpu().tolist()
if showResults:
    visualizeBoxes(imageDetection, detectedBoxes, 'Rezultat detekcije')

imagesClassification = []
pilImage = Image.fromarray(imageDetection)
for box in detectedBoxes:
    crp = getCroppedImage(pilImage, box)
    imagesClassification.append(crp)

# Classification
classInput = torch.stack(imagesClassification, 0).to(device=torch.device("cuda", 0))
output = classModel(classInput)
pred = output.argmax(dim=1)

# Segmentation
output[:,pred[:]].sum().backward()

gradients = classModel.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
activations = classModel.get_activations(classInput).detach() # num_imagesx1024x7x7
for i in range(1024):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze().cpu()
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
heatmap = heatmap.numpy()

if showResults:
    plt.figure()
    plt.suptitle('Rezultat klasifikacije')
    plt.imshow(imageDetection)
    for i in range(len(imagesClassification)):
        heatmap1 = centerCropImage(heatmap[i], scale*detectedBoxes[i][2], scale*detectedBoxes[i][3])
        segmentated = heatmap1 > 0.55*heatmap1.max()
        plt.imshow(np.flipud(segmentated), alpha=0.8, cmap = matplotlib.colors.ListedColormap([(0,0,0,0), cmaplist[clsmap[int(pred[i])]]]),
                extent=(scale*(detectedBoxes[i][0] - detectedBoxes[i][2]/2), scale*(detectedBoxes[i][0] + detectedBoxes[i][2]/2), 
                        scale*(detectedBoxes[i][1] - detectedBoxes[i][3]/2), scale*(detectedBoxes[i][1] + detectedBoxes[i][3]/2)))
        plt.text(scale*detectedBoxes[i][0], scale*detectedBoxes[i][1], classes[clsmap[int(pred[i])]], fontsize=10, horizontalalignment='center', 
                verticalalignment='center', color='w', bbox=dict(facecolor=cmaplist[clsmap[int(pred[i])]], edgecolor=cmaplist[clsmap[int(pred[i])]]))
    plt.xlim(0, width*scale)
    plt.ylim(height*scale, 0)
    plt.show()

# Measure performance
end = time.time() - start
print(f'End to end detection finished in {end}s)')

