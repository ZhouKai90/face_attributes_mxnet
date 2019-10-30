"""
Generate base anchors on index 0
"""
from __future__ import print_function
import sys
from builtins import range
import numpy as np
from ..cython.anchors import anchors_cython


def anchors_plane(feat_h, feat_w, stride, base_anchor):
	return anchors_cython(feat_h, feat_w, stride, base_anchor)

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
					 scales=2 ** np.arange(3, 6)):
	"""
	Generate anchor (reference) windows by enumerating aspect ratios X
	scales wrt a reference (0, 0, 15, 15) window.
	base_size:指定了最初的感受野的区域大小，多层卷积之后，feature_map上的一个cell的
			感受野对应于原始图片的区域大小（例如１６＊１６）
	ratios:生成的anchors在base_size上需要乘以的长宽比
	sacles:生成的anchors需要进行的倍数放大

	注意对应于原论文，base_size经过ratios变换后会得到三个不同长宽比的anchor,scale是在这三个不同比例的anchor上，
	所以最后得到的anchor个数应该是3^2=9
	最后得到的anchor的坐标都是相对于原图的？？？？
	"""
	#表示一个base_size的区域，四个值表示左上角和右下角的坐标点
	base_anchor = np.array([1, 1, base_size, base_size]) - 1
	#将base_size的区域进行ratio变化，输出多种宽高比的anchors
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	#将base_size进行scale变换
	anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
						 for i in range(ratio_anchors.shape[0])])
	return anchors

def generate_anchors_fpn(base_size=[64,32,16,8,4], ratios=[0.5, 1, 2], scales=8):
   """
   Generate anchor (reference) windows by enumerating aspect ratios X
   scales wrt a reference (0, 0, 15, 15) window.
   """
   anchors = []
   _ratios = ratios.reshape((len(base_size), -1))
   _scales = scales.reshape((len(base_size), -1))
   for i,bs in enumerate(base_size):
     __ratios = _ratios[i]
     __scales = _scales[i]
     #print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
     r = generate_anchors(bs, __ratios, __scales)
     #print('anchors_fpn', r.shape, file=sys.stderr)
     anchors.append(r)
   return anchors

def generate_anchors_dense(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6), ctr_offsets=[[0.5]]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors_list = []
    if len(scales.shape) == 0:
        scales = np.array([scales])
    for i in range(len(scales)):
        scale = scales[i]
        ctr_offset = ctr_offsets[i]
        for y_ctr_offset in ctr_offset:
            for x_ctr_offset in ctr_offset:
                ## only one ratio_anchors , ratio = 1.0 for face
                anchors_list.append(_scale_enum_dense(ratio_anchors[0, :], scale, x_ctr_offset, y_ctr_offset))
    anchors = np.concatenate(anchors_list)
    return anchors


# def generate_anchors_fpn():
# 	"""
# 	Generate anchor (reference) windows by enumerating aspect ratios X
# 	scales wrt a reference (0, 0, 15, 15) window.
# 	"""
# 	anchors = []
# 	for k, v in config.RPN_ANCHOR_CFG.items():
# 		bs = v['BASE_SIZE']
# 		__ratios = np.array(v['RATIOS'])
# 		__scales = np.array(v['SCALES'])
# 		#print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
# 		r = generate_anchors(bs, __ratios, __scales)
# 		#print('anchors_fpn', r.shape, file=sys.stderr)
# 		anchors.append(r)
#
# 	return anchors

def _whctrs(anchor):
	"""
	Return width, height, x center, and y center for an anchor (window).
	"""

	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5 * (w - 1)
	y_ctr = anchor[1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr

def _whctrs_dense(anchor, x_ctr_offset, y_ctr_offset):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + x_ctr_offset * (w - 1)
    y_ctr = anchor[1] + y_ctr_offset * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
	"""
	Given a vector of widths (ws) and heights (hs) around a center
	(x_ctr, y_ctr), output a set of anchors (windows).
	"""

	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
						 y_ctr - 0.5 * (hs - 1),
						 x_ctr + 0.5 * (ws - 1),
						 y_ctr + 0.5 * (hs - 1)))
	return anchors


def _ratio_enum(anchor, ratios):
	"""
	Enumerate a set of anchors for each aspect ratio wrt an anchor.
	将ratios的值叠加到anchor上
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor)        #获取anchors的中心点坐标和宽高
	size = w * h                                #size:16*16=256
	size_ratios = size / ratios                 #256/ratios[0.5,1,2]=[512,256,128] ???为啥要除以，sacle不是才应该这样操作么？
	ws = np.round(np.sqrt(size_ratios))         #round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
	hs = np.round(ws * ratios)                  #ws和hs进行比例计算
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors


def _scale_enum(anchor, scales):
	"""
	Enumerate a set of anchors for each scale wrt an anchor.
	"""
	w, h, x_ctr, y_ctr = _whctrs(anchor)
	# if len(scales.shape) == 0:
	# 	scales = np.array([scales])
	ws = w * scales
	hs = h * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def _scale_enum_dense(anchor, scales, x_ctr_offset, y_ctr_offset):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs_dense(anchor, x_ctr_offset, y_ctr_offset)
    if len(scales.shape) == 0:
        scales = np.array([scales])
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors