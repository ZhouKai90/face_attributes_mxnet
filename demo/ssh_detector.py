from __future__ import print_function
import cv2
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, kpoint_pred, clip_points
from rcnn.processing.generate_anchor import *
from rcnn.processing.nms import gpu_nms_wrapper
import datetime

RPN_ANCHOR_CFG = {
    '32': {'SCALES': (16, 8), 'BASE_SIZE': 32, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 64, 'CENTER_OFFSET':[[0.5],[0.5]]},
    '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 32, 'CENTER_OFFSET':[[0.5],[0.5]]},
    '8': {'SCALES': (4, 2), 'BASE_SIZE': 8, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 16, 'CENTER_OFFSET':[[0.5],[0.5]]},
}

class SSHDetector:
    def __init__(self, prefix, epoch, ctx_id=1, nms_threshold=0.3, test_mode=False):
        self.ctx_id = ctx_id
        self.ctx = mx.gpu(self.ctx_id)
        self.keys = []
        strides = []
        base_size = []
        scales = []
        self.feat_strides = [32, 16, 8]

        for s in self.feat_strides:
            self.keys.append('stride%s' % s)
            strides.append(int(s))
            base_size.append(RPN_ANCHOR_CFG[str(s)]['BASE_SIZE'])
            scales += RPN_ANCHOR_CFG[str(s)]['SCALES']

        # self._scales = np.array([32, 16, 8, 4, 2, 1])
        self._scales = np.array(scales)
        self._ratios = np.array([1.0]*len(self.feat_strides))
        # self._anchors_fpn = dict(list(zip(self.keys, generate_anchors_fpn())))
        self._anchors_fpn = dict(list(zip(self.keys, generate_anchors_fpn(base_size=base_size, scales=self._scales, ratios=self._ratios))))

        self._num_anchors = dict(zip(self.keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self._rpn_pre_nms_top_n = 1000
        #self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self.nms_threshold = nms_threshold      #值越大，同一个人脸产生的预测框越多
        self._bbox_pred = nonlinear_pred
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        self.pixel_means = np.array([103.939, 116.779, 123.68]) #BGR
        self.pixel_means = np.array([127., 127., 127.])

        if not test_mode:
            image_size = (640, 640)
            self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
            self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
            self.model.set_params(arg_params, aux_params)
        else:
            from rcnn.core.module import MutableModule
            image_size = (2400, 2400)
            data_shape = [('data', (1, 3, image_size[0], image_size[1]))]
            self.model = MutableModule(symbol=sym, data_names=['data'], label_names=None,
                                       context=self.ctx, max_data_shapes=data_shape)
            self.model.bind(data_shape, None, for_training=False)
            self.model.set_params(arg_params, aux_params)


    def detect(self, img, threshold=0.5, scales=[1.0]):
        proposals_list = []
        proposals_kp_list = []
        scores_list = []
        print('detect shape', img.shape)
        for im_scale in scales:
            if im_scale != 1.0:
                im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            else:
                im = img
            im = im.astype(np.float32)
            # im_shape = im.shape
            # self.model.bind(data_shapes=[('data', (1, 3, im_shape[0], im_shape[1]))], for_training=False)
            im_info = [im.shape[0], im.shape[1], im_scale]
            im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
            for i in range(3):
                im_tensor[0, i, :, :] = im[:, :, 2 - i] - self.pixel_means[2 - i] #bgr2rgb  mxnet rgb  opencv bgr and (h, w, c) to (c, h, w)
            data = nd.array(im_tensor)
            db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
            
            timea = datetime.datetime.now()
            self.model.forward(db, is_train=False)
            timeb = datetime.datetime.now()
            diff = timeb - timea
            print('forward uses', diff.total_seconds(), 'seconds')

            net_out = self.model.get_outputs()      #网络的输出为len=9的list,针对三个不同的stride,分为三大块的list,其中每个list分别代表score,bbox,kpoint三个维度的结果，
            pre_nms_topN = self._rpn_pre_nms_top_n
            #post_nms_topN = self._rpn_post_nms_top_n
            #min_size_dict = self._rpn_min_size_fpn

            for s in self.feat_strides:
                _key = 'stride%s' % s
                # print(_key)
                stride = int(s)
                if s == self.feat_strides[0]:
                    idx = 0
                if s == self.feat_strides[1]:
                    idx = 3
                elif s == self.feat_strides[2]:
                    idx = 6
                # print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
                scores = net_out[idx].asnumpy()     #获取每个stride下的分类得分

                idx += 1
                # print('scores',stride, scores.shape, file=sys.stderr)
                scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]    #去掉了其中lable的值？？？
                bbox_deltas = net_out[idx].asnumpy()
                idx += 1
                _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
                height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                # kpoint
                kpoint_deltas = net_out[idx].asnumpy()

                A = self._num_anchors['stride%s' % s]
                K = height * width
                anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s' % s].astype(np.float32))       #RP映射回原图中的坐标位置
                # print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
                anchors = anchors.reshape((K * A, 4))

                # print('predict bbox_deltas', bbox_deltas.shape, height, width)
                bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                # print('after clip pad', bbox_deltas.shape, height, width)
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

                kpoint_deltas = self._clip_pad(kpoint_deltas, (height, width))
                kpoint_deltas = kpoint_deltas.transpose((0, 2, 3, 1)).reshape((-1, 10))

                scores = self._clip_pad(scores, (height, width))
                scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

                # print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
                proposals = self._bbox_pred(anchors, bbox_deltas)
                proposals = clip_boxes(proposals, im_info[:2])  #将超出图像的坐标去除掉

                proposals_kp = kpoint_pred(anchors, kpoint_deltas)
                proposals_kp = clip_points(proposals_kp, im_info[:2])
                #取出score的top N
                scores_ravel = scores.ravel()
                order = scores_ravel.argsort()[::-1]
                if pre_nms_topN > 0:
                    order = order[:pre_nms_topN]
                proposals = proposals[order, :]
                proposals_kp = proposals_kp[order, :]
                scores = scores[order]

                proposals /= im_scale
                proposals_kp /= im_scale

                proposals_list.append(proposals)
                proposals_kp_list.append(proposals_kp)
                scores_list.append(scores)

        proposals = np.vstack(proposals_list)
        proposals_kp = np.vstack(proposals_kp_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        #if config.TEST.SCORE_THRESH>0.0:
        #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
        #  order = order[:_count]
        #if pre_nms_topN > 0:
        #    order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        proposals_kp = proposals_kp[order, :]
        scores = scores[order]

        det = np.hstack((proposals, scores, proposals_kp)).astype(np.float32)

        #if np.shape(det)[0] == 0:
        #    print("Something wrong with the input image(resolution is too low?), generate fake proposals for it.")
        #    proposals = np.array([[1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)
        #    scores = np.array([[0.9]]*post_nms_topN, dtype=np.float32)
        #    det = np.array([[1.0, 1.0, 2.0, 2.0, 0.9]]*post_nms_topN, dtype=np.float32)


        if self.nms_threshold < 1.0:
            keep = self.nms(det)
            det = det[keep, :]
        if threshold > 0.0:
            keep = np.where(det[:, 4] >= threshold)[0]
            det = det[keep, :]
        return det

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor
