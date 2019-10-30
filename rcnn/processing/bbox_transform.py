import numpy as np
from rcnn.cython.bbox import bbox_overlaps_cython
from ..cython.bbox import bbox_overlaps_cython


def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_cython(boxes, query_boxes)


def bbox_overlaps_py(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_points(points, im_shape):

    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    points[:, 0::10] = np.maximum(np.minimum(points[:, 0::10], im_shape[1] - 1), 0)
    points[:, 1::10] = np.maximum(np.minimum(points[:, 1::10], im_shape[0] - 1), 0)
    points[:, 2::10] = np.maximum(np.minimum(points[:, 2::10], im_shape[1] - 1), 0)
    points[:, 3::10] = np.maximum(np.minimum(points[:, 3::10], im_shape[0] - 1), 0)
    points[:, 4::10] = np.maximum(np.minimum(points[:, 4::10], im_shape[1] - 1), 0)
    points[:, 5::10] = np.maximum(np.minimum(points[:, 5::10], im_shape[0] - 1), 0)
    points[:, 6::10] = np.maximum(np.minimum(points[:, 6::10], im_shape[1] - 1), 0)
    points[:, 7::10] = np.maximum(np.minimum(points[:, 7::10], im_shape[0] - 1), 0)
    points[:, 8::10] = np.maximum(np.minimum(points[:, 8::10], im_shape[1] - 1), 0)
    points[:, 9::10] = np.maximum(np.minimum(points[:, 9::10], im_shape[0] - 1), 0)

    return points

def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
    #将(x1,y1) (x2,y2)转换为(x,y) (w,h)这种表示方法
    #实现论文loss中关于bounding box regression
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def kpoint_transform(ex_rois, kp_rois):
    """
    compute  regression targets from ex_rois to kp_rois
    :param ex_rois: [N, 4]
    :param kp_rois: [N, 10]
    :return: [N, 10]
    计算RP的中心点和keyPiont中各个点之间的相对偏移量作为target
    这个部分要好好理解，这只是taget,在实际训练的时候的output需要和这个target做Smooth L1
    """
    assert ex_rois.shape[0] == kp_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    targets_p1_dx = (kp_rois[:, 0] - ex_ctr_x) / (ex_widths + 1e-14)
    targets_p2_dx = (kp_rois[:, 2] - ex_ctr_x) / (ex_widths + 1e-14)
    targets_p3_dx = (kp_rois[:, 4] - ex_ctr_x) / (ex_widths + 1e-14)
    targets_p4_dx = (kp_rois[:, 6] - ex_ctr_x) / (ex_widths + 1e-14)
    targets_p5_dx = (kp_rois[:, 8] - ex_ctr_x) / (ex_widths + 1e-14)
    targets_p1_dy = (kp_rois[:, 1] - ex_ctr_y) / (ex_heights + 1e-14)
    targets_p2_dy = (kp_rois[:, 3] - ex_ctr_y) / (ex_heights + 1e-14)
    targets_p3_dy = (kp_rois[:, 5] - ex_ctr_y) / (ex_heights + 1e-14)
    targets_p4_dy = (kp_rois[:, 7] - ex_ctr_y) / (ex_heights + 1e-14)
    targets_p5_dy = (kp_rois[:, 9] - ex_ctr_y) / (ex_heights + 1e-14)

    targets = np.vstack(
        (targets_p1_dx, targets_p1_dy, targets_p2_dx, targets_p2_dy, targets_p3_dx, targets_p3_dy, targets_p4_dx, targets_p4_dy, targets_p5_dx, targets_p5_dy)).transpose()
    return targets

def nonlinear_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]
	#网络的输出是用来做bounding box regression的target,所以要根据公式计算出预测的坐标位置
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
	#将(center_x, cnter_y, w, h)转换为(x1, y1, x2, y2)
    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def kpoint_pred(boxes, point_deltas):
    """
    Transform the set of class-agnostic key point into class-specific key point
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, point_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    d1x = point_deltas[:, 0]
    d1y = point_deltas[:, 1]
    d2x = point_deltas[:, 2]
    d2y = point_deltas[:, 3]
    d3x = point_deltas[:, 4]
    d3y = point_deltas[:, 5]
    d4x = point_deltas[:, 6]
    d4y = point_deltas[:, 7]
    d5x = point_deltas[:, 8]
    d5y = point_deltas[:, 9]

    pred_points = np.zeros(point_deltas.shape)

    # print("aa", d1x.shape, widths.shape, ctr_x.shape, x.shape)
    pred_points[:, 0] = d1x * widths + ctr_x
    pred_points[:, 1] = d1y * heights + ctr_y
    pred_points[:, 2] = d2x * widths + ctr_x
    pred_points[:, 3] = d2y * heights + ctr_y
    pred_points[:, 4] = d3x * widths + ctr_x
    pred_points[:, 5] = d3y * heights + ctr_y
    pred_points[:, 6] = d4x * widths + ctr_x
    pred_points[:, 7] = d4y * heights + ctr_y
    pred_points[:, 8] = d5x * widths + ctr_x
    pred_points[:, 9] = d5y * heights + ctr_y

    return pred_points

def iou_transform(ex_rois, gt_rois):
    """ return bbox targets, IoU loss uses gt_rois as gt """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
    return gt_rois


def iou_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    dx1 = box_deltas[:, 0::4]
    dy1 = box_deltas[:, 1::4]
    dx2 = box_deltas[:, 2::4]
    dy2 = box_deltas[:, 3::4]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = dx1 + x1[:, np.newaxis]
    # y1
    pred_boxes[:, 1::4] = dy1 + y1[:, np.newaxis]
    # x2
    pred_boxes[:, 2::4] = dx2 + x2[:, np.newaxis]
    # y2
    pred_boxes[:, 3::4] = dy2 + y2[:, np.newaxis]

    return pred_boxes


# define bbox_transform and bbox_pred
bbox_transform = nonlinear_transform
bbox_pred = nonlinear_pred
