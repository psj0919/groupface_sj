import os
import random

from PIL import Image
import numpy as np
import torchvision
import cv2


def Argumentation_RandomNoise(image, noise_num=300):
    img_noise = image
    rows, cols, chn = img_noise.shape
    # ???
    for i in range(noise_num):
        x = np.random.randint(0, rows)  # ???????????
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def Argumentation_SaltPepperNoise(image, prob=0.003):
    output = np.zeros(image.shape, np.uint8)
    noise_out = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]
                noise_out[i][j] = 100
    return output


def Argumentation_GasussNoise(image, mean=0, var=1e-2):
    image = np.array(image / 255, dtype=float)
    mean = np.mean(image)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    noise = noise * 255
    return out


def Argumentation_MotionBlur(img):
    random_s = int(np.random.normal(1, 0.1, 1)[0] * 10)
    if random_s < 0:
        random_s = 0
    random_size = int(np.random.normal(1, 0.5, 1)[0] * 50)
    if random_size < 40:
        random_size = 40
    random_degree = int(np.random.uniform(-180, 180, 1)[0])

    size = random_s
    img_resize = random_size
    degree = random_degree

    center = (size // 2, size // 2)

    M = cv2.getRotationMatrix2D(center, degree, 1.0)

    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    if np.random.normal(0, 0.1, 1)[0] < 0:
        kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (size, size))

    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output


def Argumentation_HorizontalFlip(img):
    horizon_flip = torchvision.transforms.RandomHorizontalFlip(1.0)
    img = horizon_flip(Image.fromarray(img))
    return np.array(img)


def Argumentation_CenterCrop(img):
    ratio = 0.83
    h = img.shape[0]
    w = img.shape[1]
    center_cro = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((int(h * ratio), int(w * ratio))),
        torchvision.transforms.Resize((h, w))
    ])
    img = center_cro(Image.fromarray(img))
    return np.array(img)


def Argumentation_randomRotate(img):
    rotate = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation((-5, 5))
    ])
    img = rotate(Image.fromarray(img))
    return np.array(img)


def Argumentation_randomAffine(img):
    affine = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])
    img = affine(Image.fromarray(img))
    return np.array(img)


def Argumentation_ColorJit(img):
    colorJit = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3)
    img = colorJit(Image.fromarray(img))
    return np.array(img)


def Argumentation_RandomBlack(image, block_size=30, block_num=4):
    for i in range(random.randint(1, block_num)):
        c_x = random.randint(0, image.shape[1])
        c_y = random.randint(0, image.shape[0])
        size = random.randint(0, block_size)

        start_x = max(0, c_x - int(size / 2))
        end_x = min(image.shape[1], c_x + int(size / 2))

        start_y = max(0, c_y - int(size / 2))
        end_y = min(image.shape[0], c_y + int(size / 2))

        image[start_y:end_y, start_x:end_x] = 0
    return image


def ArgumentationSchedule(img, seed):
    # print("choosing seed:{}".format(seed))
    fun = {
        0: Argumentation_MotionBlur,
        1: Argumentation_randomRotate,
        2: Argumentation_CenterCrop,
        3: Argumentation_ColorJit,
        4: Argumentation_HorizontalFlip,
        5: Argumentation_randomAffine,
        6: Argumentation_RandomBlack,
        7: Argumentation_RandomNoise,
        8: Argumentation_SaltPepperNoise,
        9: Argumentation_GasussNoise,
    }.get(seed)
    return fun(img)


def case_testLoader():
    while True:
        # img = Image.open("../../ims/head1_17.jpg")
        img = cv2.imread("../../ims/head1_17.jpg")
        img = ArgumentationSchedule(np.array(img), random.randint(0, 9))
        img = np.array(img, dtype=np.uint8)
        # cv2.imshow("dataSetWindow", img[:, :, ::-1])
        cv2.imshow("dataSetWindow", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    case_testLoader()
