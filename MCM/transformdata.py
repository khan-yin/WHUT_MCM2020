from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms



img_size = 32


def get_train_transforms():
    return transforms.Compose([
        # transforms.RandomResizedCrop((img_size, img_size)),
        # 1 CenterCrop
        transforms.CenterCrop(img_size),     # 512

        # 2 RandomCrop
        # transforms.RandomCrop(224, padding=16),
        # transforms.RandomCrop(224, padding=(16, 64)),
        # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
        # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
        # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
        # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
        # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

        # 3 RandomResizedCrop
        transforms.RandomResizedCrop(size=img_size, scale=(0.5, 0.5)),

        # 4 FiveCrop
        # transforms.FiveCrop(112),
        # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

        # 5 TenCrop
        # transforms.TenCrop(112, vertical_flip=False),
        # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

        # 1 Horizontal Flip
        transforms.RandomHorizontalFlip(p=0.75),

        # 2 Vertical Flip
        transforms.RandomVerticalFlip(p=0.5),

        # 3 RandomRotation
        transforms.RandomRotation(90),
        # transforms.RandomRotation((90), expand=True),
        # transforms.RandomRotation(30, center=(0, 0)),
        # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # return Compose([
    #         #将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
    #         RandomResizedCrop(img_size, img_size),
    #         # 转置
    #         Transpose(p=0.5),
    #         HorizontalFlip(p=0.5), #img翻转
    #         VerticalFlip(p=0.5),# 依据概率p对PIL图片进行垂直翻转
    #         ShiftScaleRotate(p=0.5),# 随机放射变换（ShiftScaleRotate），该方法可以对图片进行平移（translate）、缩放（scale）和旋转（roatate）
    #         # 随机改变图片的 HUE、饱和度和值
    #         HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    #         #随机亮度对比度
    #         RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    #         #将像素值除以255 = 2 ** 40 - 1，减去每个通道的平均值并除以每个通道的std
    #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    #         # 在图像上生成矩形区域。
    #         CoarseDropout(p=0.5),
    #         # 在图像中生成正方形区域。
    #         Cutout(p=0.5),
    #         ToTensorV2(p=1.0),
    #     ], p=1.)

def get_valid_transforms():
    return transforms.Compose([
        transforms.CenterCrop(img_size),  # 512
        transforms.RandomResizedCrop((img_size, img_size)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # return Compose([
    #     CenterCrop(img_size,img_size,p=1.),
    #     Resize(img_size,img_size),
    #     Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],max_pixel_value=255.0,p=1.0),
    #     ToTensorV2(p=1.0),
    # ],p=1.)


def get_test_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # return Compose([
    #     #随机裁剪大小
    #     RandomResizedCrop(img_size,img_size),
    #     #转置
    #     Transpose(p=0.5),
    #     #翻转
    #     HorizontalFlip(p=0.5),
    #     #垂直翻转
    #     VerticalFlip(p=0.5),
    #     #随机色调，饱和度
    #     HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    #     #随机亮度对比度
    #     RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    #     #Normalize归一化
    #     Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],max_pixel_value=255.0,p=1.0),
    #     ToTensorV2(p=1.),
    # ],p=1.)
