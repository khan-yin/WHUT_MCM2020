import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
import os
# 加载模型
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

# 导入所需的图像预处理模块
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
path = glob.glob('./origin/Negative/*.jpg')
c = 0
par = tqdm(enumerate(path),total=len(path))
for i,filename in par:
    # filename = 'D:\python\MCM\origin\Positive\ATT1_DSCN9647.jpg'
    newname = filename.split('\\')[1]
    # print(newname)
    isgood = False
    # 将图片输入到网络之前执行预处理
    '''
    1、加载图像，load_img
    2、将图像从PIL格式转换为Numpy格式，image_to_array
    3、将图像形成批次，Numpy的expand_dims
    '''
    # 以PIL格式加载图像
    original = load_img(filename, target_size=(224, 224))
    # print('PIL image size', original.size)
    # plt.imshow(original)
    # plt.show()

    # 将输入图像从PIL格式转换为Numpy格式
    # In PIL-- 图像为（width, height, channel）
    # In Numpy——图像为（height, width, channel）
    numpy_image = img_to_array(original)
    # plt.imshow(np.uint8(numpy_image))
    # plt.show()
    # print('numpy array size', numpy_image.size)

    # 将图像/图像转换为批量格式
    # expand_dims将为特定轴上的数据添加额外的维度
    # 网络的输入矩阵具有形式（批量大小，高度，宽度，通道）
    # 因此，将额外的维度添加到轴0。
    image_batch = np.expand_dims(numpy_image, axis=0)
    # print('image batch size', image_batch.shape)
    # plt.imshow(np.uint8(image_batch[0]))

    # 使用各种网络进行预测
    # 通过从批处理中的图像的每个通道中减去平均值来预处理输入。
    # 平均值是通过从ImageNet获得的所有图像的R，G，B像素的平均值获得的三个元素的阵列
    # 获得每个类的发生概率
    # 将概率转换为人类可读的标签
    # VGG16 网络模型
    # 对输入到VGG模型的图像进行预处理
    processed_image_vgg = vgg16.preprocess_input(image_batch.copy())
    processed_image_resnet = resnet50.preprocess_input(image_batch.copy())
    processed_image_mobilenet = mobilenet.preprocess_input(image_batch.copy())
    # processed_image_inception_v3 = inception_v3.preprocess_input(image_batch.copy())
    # 获取预测得到的属于各个类别的概率
    predictions_vgg = vgg_model.predict(processed_image_vgg)
    predictions_resnet = resnet_model.predict(processed_image_resnet)
    predictions_mobilenet = mobilenet_model.predict(processed_image_mobilenet)
    # predictions_inception_v3 = inception_model.predict(processed_image_inception_v3)

    # 获取预测得到属于各个类别的概率



    # 获取预测得到属于各个类别的概率



    # 获取预测得到的属于各个类别的概率

    # print(predictions.shape)
    # print(predictions[:,309]>0.60)
    res_vgg = predictions_vgg[:,309]
    res_resnet = predictions_resnet[:, 309]
    res_mobilenet = predictions_mobilenet[:, 309]
    # res_inception_v3 = predictions_inception_v3[:, 309]

    # print(label_vgg)
    p = 0.55
    dirs = './newdata/55/Negative/'
    isExists = os.path.exists(dirs)
    if not isExists:
        os.makedirs(dirs)
    if res_vgg>p or res_resnet>p or res_mobilenet>p :
        isgood=True
        print('true')
    # 输出预测值
    # 将预测概率转换为类别标签
    # 缺省情况下将得到最有可能的五种类别


    # ResNet50网络模型
    # 对输入到ResNet50模型的图像进行预处理
    # processed_image = resnet50.preprocess_input(image_batch.copy())

    # 获取预测得到的属于各个类别的概率
    # predictions = resnet_model.predict(processed_image)

    # 将概率转换为类标签
    # 如果要查看前3个预测，可以使用top参数指定它
    # label_resnet = decode_predictions(predictions, top=3)
    # label_resnet

    # MobileNet网络结构
    # 对输入到MobileNet模型的图像进行预处理
    # processed_image = mobilenet.preprocess_input(image_batch.copy())

    # 获取预测得到属于各个类别的概率
    # predictions = mobilenet_model.predict(processed_image)

    # 将概率转换为类标签
    # label_mobilnet = decode_predictions(predictions)
    # label_mobilnet

    # InceptionV3网络结构
    # 初始网络的输入大小与其他网络不同。 它接受大小的输入（299,299）。
    # 因此，根据它加载具有目标尺寸的图像。
    # 加载图像为PIL格式
        original = load_img(filename, target_size=(299, 299))

        # 将PIL格式的图像转换为Numpy数组
        numpy_image = img_to_array(original)

        # 根据批量大小重塑数据
        image_batch = np.expand_dims(numpy_image, axis=0)

        # 将输入图像转换为InceptionV3所能接受的格式
        # processed_image = inception_v3.preprocess_input(image_batch.copy())

        # 获取预测得到的属于各个类别的概率
        # predictions = inception_model.predict(processed_image)

        # 将概率转换为类标签
        # label_inception = decode_predictions(predictions)
        # label_inception

        import cv2

        numpy_image = np.uint8(img_to_array(original)).copy()
        numpy_image = cv2.resize(numpy_image, (900, 900))

        # cv2.putText(numpy_image, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]), (350, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # cv2.putText(numpy_image, "MobileNet: {}, {:.2f}".format(label_mobilenet[0][0][1], label_mobilenet[0][0][2]), (350, 75),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # cv2.putText(numpy_image, "Inception: {}, {:.2f}".format(label_inception[0][0][1], label_inception[0][0][2]), (350, 110),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # cv2.putText(numpy_image, "ResNet50: {}, {:.2f}".format(label_resnet[0][0][1], label_resnet[0][0][2]), (350, 145),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        numpy_image = cv2.resize(numpy_image, (700, 700))
        # print('/newimg/'+newname)
        cv2.imwrite(dirs+str(i)+'.jpg',
                    cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

        # plt.figure(figsize=[10, 10])
        # plt.imshow(numpy_image)
        # plt.axis('off')