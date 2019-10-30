# -*- coding: UTF-8 -*-
import cv2
import os
import argparse
import numpy as np
from fa_predictor import FaceAttributesPredictor
from ssh_detector import SSHDetector
from face_align import norm_crop
import random

interestAttribute = [
        'Gender',
        'Mask',
        'Glass',
        'MouthOpen',
        'EyesClose'
        ]

def get_images_name_list(directory):
    name_list = []
    assert directory is not None
    for root, dirs, files in os.walk(directory, topdown = True):
        for name in files:
            print(os.path.join(root, name))
            name_list.append(os.path.join(root, name))
    return name_list

def save_crop_face(faces, imgName, imgMat):
    savePath = './output'
    imgName = imgName.split('/')[-1]
    print(imgName)
    for num in range(faces.shape[0]):
        bbox = faces[num, 0:4]
        cv2.rectangle(imgMat, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        kpoint = faces[num, 5:15]
        for knum in range(5):
            cv2.circle(imgMat, (kpoint[2 * knum], kpoint[2 * knum + 1]), 2, [0, 0, 255], 2)
    cv2.imwrite(os.path.join(savePath, imgName), imgMat)


def crop_face(faces, imgName, imgMat):
    savePath = './images'
    imgName = imgName.split('/')[-1]
    index = 0
    print(imgMat.shape)
    for num in range(faces.shape[0]):
        kpoint = faces[num, 5:15]
        kpoint = np.array(kpoint).reshape(5,2)
        cropImg = norm_crop(imgMat, kpoint)
        cv2.imwrite(os.path.join(savePath, '{}_{}'.format(index,imgName)), cropImg)
        index += 1

def show_face_rec_kpoint_attribute(faces, imgName, imgMat, faceAttributeList):
    savePath = './images'
    imgName = imgName.split('/')[-1]
    print(imgName)
    for num in range(faces.shape[0]):
        bbox = faces[num, 0:4]
        cv2.rectangle(imgMat, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        kpoint = faces[num, 5:15]
        for knum in range(5):
            cv2.circle(imgMat, (kpoint[2 * knum], kpoint[2 * knum + 1]), 2, [0, 0, 255], 2)
        faceAttribute = faceAttributeList[num]
        text = 'Gender:{}\nMask:{}\nGlass:{}\nMouthOpen:{}\nEyesClose:{}\n'.format(faceAttribute['Gender'], faceAttribute['Mask'],
                                                                                  faceAttribute['Glass'], faceAttribute['MouthOpen'], faceAttribute['EyesClose'])
        colour = (random.randint(0, 255),  random.randint(0, 255), random.randint(0, 255))
        for i, txt in enumerate(text.split('\n')):
            cv2.putText(imgMat, txt, (bbox[2], int(bbox[1]+i*20)), 1, 2, colour, 2)
    cv2.imwrite(os.path.join(savePath, imgName), imgMat)


def test(FAPrefix, SSHPrefix, imgPath):
    FAPredictor = FaceAttributesPredictor(FAPrefix, epoch=1, ctxId=0)
    faceDetector = SSHDetector(SSHPrefix, epoch=0, ctx_id=0)
    images_name = get_images_name_list(imgPath)
    for img in images_name:
        print('image: %s' % (img))
        imgMat = cv2.imread(img)
        faces = faceDetector.detect(imgMat, threshold=0.8)
        print(faces.shape[0], ' faces detected.\n')
        imgMatList = []
        for num in range(faces.shape[0]):
            kpoint = faces[num, 5:15]
            kpoint = np.array(kpoint).reshape(5, 2)
            cropImg = norm_crop(imgMat, kpoint)
            imgMatList.append(cropImg)
        faResult = FAPredictor.predict(imgMatList)
        assert len(faResult) ==len(imgMatList)
        show_face_rec_kpoint_attribute(faces, img, imgMat, faResult)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for multi-labels face attribute detection.')
    parser.add_argument('image_path', help='path of image to test.')
    args = parser.parse_args()

    FAPrefix = os.getcwd() + '/model/eighth_resmobile'
    SSHPrefix = os.getcwd() + '/model/symbol_ssh'

    test(FAPrefix=FAPrefix, SSHPrefix=SSHPrefix, imgPath=args.image_path)