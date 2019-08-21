# -*- coding: UTF-8 -*-
import mxnet as mx
import cv2
import os
import argparse
from collections import namedtuple
import numpy as np

interestAttribute = [
        'Gender',
        'Mask',
        'Glass',
        'MouthOpen',
        'EyesOpen'
        ]

def get_images_name_list(directory):
    name_list = []
    assert directory is not None
    for root, dirs, files in os.walk(directory, topdown = True):
        for name in files:
            print(os.path.join(root, name))
            name_list.append(os.path.join(root, name))
    return name_list

def load_single_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        # logger.error('imread imges failed')
        return None
    #mxnet三通道输入图像是严格的RGB格式，而cv2读入的是BGR格式，需要进行转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #注意输出的是(width, height)
    img = cv2.resize(img, (112, 112))
    (dirs, ext) = os.path.splitext(img_path)
    # cv2.imwrite('{}_{}_{}'.format(dirs, fc.img_width, fc.img_height) + ext, img)
    #重塑图像各个纬度的信息，由[width, height, channels]转化为[channels, height, width]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    #添加batchSize第四个纬度，并创建NDArray
    img = img[np.newaxis, :]
    img = mx.nd.array(img)
    # logger.info(img.shape)
    return img

def test(module_path, path):
    ctx = mx.gpu(0)

    sym, arg_params, aux_params = mx.model.load_checkpoint(module_path, 1)

    all_layers = sym.get_internals()
    print(all_layers)
    #提取中间的某一层作为输出，在里要取softmax层之前的全连接层作为输出
    sym = all_layers['fc1_output']
    mod = mx.module.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
    mod.set_params(arg_params, aux_params)

    Batch = namedtuple('Batch', ['data'])
    images_name = get_images_name_list(path)
    for img in images_name:
        image = load_single_image(img)
        mod.forward(Batch([image]), is_train=False)

        prob = mod.get_outputs()[0].asnumpy()[0]

        print('image: %s' % (img))

        Gender_pred = np.argmax(prob[0:2])
        Mask_pred = np.argmax(prob[2:4])
        Glass_pred = np.argmax(prob[4:7])
        MouthOpen_pred = np.argmax(prob[7:10])
        EyesOpen_pred = np.argmax(prob[10:13])

        print('Gender_pred:%s' % Gender_pred)
        print('Mask_pred:%s' % Mask_pred)
        print('Glass_pred:%s' % Glass_pred)
        print('MouthOpen_pred:%s' % MouthOpen_pred)
        print('EyesOpen_pred:%s' % EyesOpen_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for multi-labels face attribute detection.')
    parser.add_argument('image_path', help='path of image to test.')
    args = parser.parse_args()

    test_model_path = '/kyle/workspace/project/face_attributes_prediction/model/resmobile'
    test(path=args.image_path, module_path=test_model_path)