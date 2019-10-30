import cv2
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd

IMG_WIDTH = 112
IMG_HIGHT = 112

class FaceAttributesPredictor:
    def __init__(self, prefix, epoch=0, ctxId=0):
        self.ctx = mx.gpu(ctxId)
        self.prefix = prefix
        self.epoch = epoch

        sym, arg_params, aux_params = mx.model.load_checkpoint(self.prefix, self.epoch)
        all_layers = sym.get_internals()
        # print(all_layers)
        # 提取中间的某一层作为输出，在里要取softmax层之前的全连接层作为输出
        sym = all_layers['fc1_output']
        self.model = mx.module.Module(symbol=sym, context=self.ctx, label_names=None)
        # self.model = mx.module.Module(symbol=sym, context=self.ctx, data_names=['data'], label_names=None)
        self.model.bind(for_training=False, data_shapes=[('data', (1, 3, IMG_WIDTH, IMG_HIGHT))])
        self.model.set_params(arg_params, aux_params)

    def predict(self, mats):
        assert len(mats) > 0
        # dataShape = (1, 3, IMG_WIDTH, IMG_HIGHT)
        # self.model.bind(for_training=False, data_shapes=[('data', dataShape)])

        result = []
        for i in range(len(mats)):
            if mats[i].shape != (IMG_WIDTH, IMG_HIGHT, 3):
                imgMat = cv2.resize(mats[i], (IMG_WIDTH, IMG_HIGHT))
            else:
                imgMat = mats[i]
            imgMat = imgMat.astype(np.float32)
            imTensor = np.zeros((1, 3, imgMat.shape[0], imgMat.shape[1]))
            for i in range(3):
                imTensor[0, i, :, :] = imgMat[:, :, 2 - i]  # bgr2rgb  mxnet rgb  opencv bgr and (h, w, c) to (c, h, w)

            data = nd.array(imTensor)
            db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])

            self.model.forward(db, is_train=False)
            print('forward')
            # probs = self.mod.get_outputs()[0].asnumpy()
            probs = self.model.get_outputs()[0].asnumpy()
            print(probs)
            for prob in probs:
                pred = {}
                pred['Gender'] = np.argmax(prob[0:2])
                pred['Mask'] = np.argmax(prob[2:4])
                pred['Glass'] = np.argmax(prob[4:7])
                pred['MouthOpen'] = np.argmax(prob[7:10])
                pred['EyesClose'] = np.argmax(prob[10:13])
                result.append(pred)

        return result