import mxnet as mx
import numpy as np
from fa_config import fa_config as fc

def acc(label, pred, label_width = fc.num_class):
    return float((label == np.round(pred)).sum()) / label_width / pred.shape[0]

def loss(label, pred):
    loss_all = 0
    for i in range(len(pred)):
        loss = 0
        loss -= label[i] * np.log(pred[i] + 1e-6) + (1.- label[i]) * np.log(1. + 1e-6 - pred[i])
        if i == 1:
            loss_all += (np.sum(loss))*2
        else:
            loss_all += np.sum(loss)
    loss_all = float(loss_all)/float(len(pred) + 0.000001)
    return  loss_all

class CustomCrossEntropyLoss(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12, name="cross-entropy-loss",
                outputName = None, inputName = None):
        super(CustomCrossEntropyLoss, self).__init__(
            name, eps=eps, output_name=outputName, input_name=inputName)
        self.eps = eps

    def update(self, labels, preds):
        preds = preds[0].asnumpy().astype('int32')
        labels = labels[0].asnumpy().astype('int32')

        firstLabel = True
        for label, pred in zip(labels, preds):
            loss = 0
            loss -= label * np.log(pred + self.eps) + (1.- label) * np.log(1. + self.eps - pred)
            if firstLabel is True:
                self.sum_metric += (np.sum(loss))*2
                firstLabel = False
            else:
                self.sum_metric += np.sum(loss)

            self.num_inst += len(pred)
        # print(self.sum_metric, self.num_inst)

class ClassAccMetric(mx.metric.EvalMetric):
    def __init__(self, name=None, label_index=0, pred_offset=0, class_num=0):
        self.axis = 1
        self.label_index = label_index
        self.pred_offset = pred_offset
        self.class_num = class_num
        super(ClassAccMetric, self).__init__(
            name=name, axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count+=1
        label = labels[0].asnumpy()[:, self.label_index:self.label_index+1]
        pred_label = preds[-1].asnumpy()[:, self.pred_offset:self.pred_offset+self.class_num]
        pred_label = np.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.astype('int32').flatten()
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

if __name__ == "__main__":

    labels = mx.nd.array([[1., 0., 0., 1., 0., 1., 0.]])
    predicts = mx.nd.array([[0.80933994, 0.10083518, 0.89632225, 0.9256714, 0.31192145, 0.80102485, 0.30593637]])

    print(loss(predicts, labels))

    CEL = CustomCrossEntropyLoss(eps = 1e-6)
    CEL.update(predicts, labels)
    loss = CEL.get()
    for i in loss:
        print(i)