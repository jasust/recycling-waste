import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_bregman

def detection_preprocessing(mode, width, height):
    image_file_paths = glob(f"data/Warp-D/{mode}/images/*")

    for i in range(len(image_file_paths)):
        image_path = image_file_paths[i]
        imgname = image_path.split('\\')[-1]
        if imgname[4] != 'z':
            image = cv2.imread(image_path)
            image2 = cv2.resize(image, (width, height))
            print(f'data/Warp-D/{mode}/images/resized/{imgname}')
            cv2.imwrite(f'data/Warp-D/{mode}/images/resized_{width}/{imgname}', image2)
    return

def segmentation_preprocessing(mode, size=224):
    image_file_paths = glob(f"data/Warp-S/{mode}_images/*")
    mask_file_paths = glob(f"data/Warp-S/{mode}_masks/*")
    for i in range(len(image_file_paths)):
        image_path = image_file_paths[i]
        mask_path = mask_file_paths[i]
        imgname = image_path.split('\\')[-1]
        maskname = mask_path.split('\\')[-1]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        h, w = image.shape[:2]
        padh = int((max(h,w)-h)/2)
        padw = int((max(h,w)-w)/2)
        image = cv2.copyMakeBorder(image.copy(), padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask.copy(), padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
        
        image2 = cv2.resize(image, (size, size))
        mask2 = cv2.resize(mask, (size, size))
        cv2.imwrite(f'data/Warp-S/{mode}_images_{size}/{imgname}', image2)
        cv2.imwrite(f'data/Warp-S/{mode}_masks_{size}/{maskname}', mask2)

        if mode == 'train':
            imgnameparts = imgname.split('.')
            masknameparts = maskname.split('.')
            image3 = cv2.flip(image2, 1)
            mask3 = cv2.flip(mask2, 1)
            cv2.imwrite(f'data/Warp-S/{mode}_images_{size}/{imgnameparts[0]}_fliph.{imgnameparts[1]}', image3)
            cv2.imwrite(f'data/Warp-S/{mode}_masks_{size}/{masknameparts[0]}_fliph.{masknameparts[1]}', mask3)
            image3 = cv2.flip(image2, 0)
            mask3 = cv2.flip(mask2, 0)
            cv2.imwrite(f'data/Warp-S/{mode}_images_{size}/{imgnameparts[0]}_flipv.{imgnameparts[1]}', image3)
            cv2.imwrite(f'data/Warp-S/{mode}_masks_{size}/{masknameparts[0]}_flipv.{masknameparts[1]}', mask3)
            image3 = cv2.flip(image2, -1)
            mask3 = cv2.flip(mask2, -1)
            cv2.imwrite(f'data/Warp-S/{mode}_images_{size}/{imgnameparts[0]}_flipb.{imgnameparts[1]}', image3)
            cv2.imwrite(f'data/Warp-S/{mode}_masks_{size}/{masknameparts[0]}_flipb.{masknameparts[1]}', mask3)
    return
    # for i in range(10): # len(image_file_paths)):
    #     image_path = image_file_paths[i]
    #     image = cv2.imread(image_path)
    #     img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #     equI = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    #     denoised = denoise_tv_bregman(image, 4)
    #     denoisedeq = denoise_tv_bregman(equI, 4)
    #     plt.figure()
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(image)
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(equI)
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(denoised)
    #     plt.subplot(1, 4, 4)
    #     plt.imshow(denoisedeq)
    #     plt.show()
    # return


if __name__ == '__main__':
    # detection_preprocessing('valid', 480, 270)
    segmentation_preprocessing('valid', 256)