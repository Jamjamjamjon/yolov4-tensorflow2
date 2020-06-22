import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from config.config import cfg
from absl import app, flags, logging
from absl.flags import FLAGS


#---------------------------
#  Function()s includes:
#
#   - load_darknet_weights() -> load darknet weights (.weights) to the model
#   - read_class_names() -> return [class_name] list
#   - get_anchors() -> return [yolov4 anchor] list
#   - image_preprocess() -> image process, shape  -> (608, 608, 3)
#   - draw_bbox()
#   - bboxes_iou()
#   - nms()
#   - detecting_info()
#   - postprocess_bbbox()
#   - postprocess_boxes()
#---------------------------



#---------------------------
#  Functions
#---------------------------


# read [class_name] file 
# return [class_name] list
def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for index, name in enumerate(data):
            names[index] = name.strip('\n')
    return names


# read [yolov4 anchors] file 
# return [yolov4 anchor] list
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = np.array(anchors.split(','), dtype=np.float32)
    
    return anchors.reshape(3, 3, 2)



# image_preporcess
# parameter: image -> must be read by OpenCV.imread()
# parameter: target_size -> 608, default
# image process, shape  -> (608, 608, 3)
def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    # image process, shape  -> (1, 608, 608, 3)
    image_paded = image_paded[np.newaxis, ...].astype(np.float32)

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes



# feature maps decode [conv_sbbox, conv_mbbox, conv_lbbox] 
# feature maps[conv_sbbox, conv_mbbox, conv_lbbox] decode 
def decode(conv_output, num_class):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))
    conv_raw_xywh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (4, 1, num_class), axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([conv_raw_xywh, pred_conf, pred_prob], axis=-1)




# decode_train
def decode_train(conv_output, num_class, strides, anchors, i=0, xyscale=[1,1,1]):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, num_class), axis=-1)

    x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0), [output_size, 1])
    y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1), [1, output_size])
    xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
    # xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * xyscale[i]) - 0.5 * (xyscale[i] - 1) + xy_grid) * strides[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)




# process bboxes shape
def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        xy_grid = np.tile(tf.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float)

        # pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        # pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    return pred_bbox


# the first time to post-process bboxes -> by socre (next would be NMS)
# confidence = P(object) * IoU 
# score = confidence * P(class(i) \ object) = p(class(i)) * IoU
def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    # scores = pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)



# IoU 
def IoU(boxes1, boxes2):
    """
    # boxes.shape - > (n,4) :[[0, 0, 3, 3], [0, 0, 9, 9], ...]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # calculate the box's area of boxes1 and boxess
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # calculate Intersection
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area


    # to ensure IoU > 0
    iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return iou


# G_IoU
def G_IoU(boxes1,boxes2):
    

    # ===========cal IOU=============#
    #cal Intersection
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # calculate the box's area of boxes1 and boxess
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # calculate Intersection
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area


    # to ensure IoU > 0
    iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)


    # ===========cal enclose area for GIOU=============#
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


# D_IoU
def D_IoU(boxes1,boxes2):
   

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    #cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])

    #cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
                 np.square(boxes1_center[...,1]-boxes2_center[...,1])

    #cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious



# C_IoU
def C_IoU(boxes1, boxes2):
   

    # calculate IoU
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    
    # get IoU
    iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    
    left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    up = np.maximum(boxes1[..., 1], boxes2[..., 1])
    right = np.maximum(boxes1[..., 2], boxes2[..., 2])
    down = np.maximum(boxes1[..., 3], boxes2[..., 3])

    # diou = d = u / c
    c = (right - left) * (right - left) + (up - down) * (up - down)
    

    ax = (boxes1[..., 0] + boxes1[..., 2]) / 2
    ay = (boxes1[..., 1] + boxes1[..., 3]) / 2
    bx = (boxes2[..., 0] + boxes2[..., 2]) / 2
    by = (boxes2[..., 1] + boxes2[..., 3]) / 2


    # diou = d = u / c
    u = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
    d = u / c

    aw = boxes1[..., 2] - boxes1[..., 0]
    ah = boxes1[..., 3] - boxes1[..., 1]
    bw = boxes2[..., 2] - boxes2[..., 0]
    bh = boxes2[..., 3] - boxes2[..., 1]

    # ar_gt =  w_gt/h_gt
    # ar_pred = w/h 
    ar_gt = bw / bh
    ar_pred = aw / ah

    # v = ar_loss = 4/pi^2 * (arctan(w_gt/h_gt) - arctan(w/h))^2
    ar_loss = 4 / (np.pi * np.pi) * (np.arctan(ar_gt) - np.arctan(ar_pred)) * (np.arctan(ar_gt) - np.arctan(ar_pred))
    
    # alpha = ar_loss / (1-IoU) + ar_loss
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)

    # ciou = diou + alpha * ar_loss
    # diou = d = u / c
    ciou_term = d + alpha * ar_loss

    ciou = iou - ciou_term

    return ciou




# NMS ->  the second time to filter the bboxes
def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            
            # iou = IoU(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            d_iou = D_IoU(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])

            
            weight = np.ones((len(d_iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = d_iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * d_iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes



