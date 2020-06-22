import cv2
import numpy as np
import time
import random
import colorsys
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import yolo_model.utils as utils
from yolo_model.yolov4 import YOLOv4
from yolo_model.utils import decode, decode_train
from PIL import Image
from config.config import cfg



#---------------------------
#  Function()s includes:
#   - image_preprocess_before_predicting() -> process the image for predicting.
#   - yolov4_model_create() -> create yolov4 model directly
#   - bboxes_optimizing() -> process the bboxes
#   - load_yolov4_model()
#   - save_yolov4_model()
#   - load video()

#---------------------------





#---------------------------
#  Functions
#---------------------------

# load darknet weights(.weights) 
def load_darknet_weights(model, weights_file):

    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(110):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [93, 101, 109]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [93, 101, 109]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    logging.info("YOLOv4 weights loaded.")


# process the image for predicting
# returns: (image_original), (image_processed), (image_original_size)
# (original_image) shape -> (w, h, 3)
# (image_processed) shape -> (1, 608, 608, 3)
def image_preprocess_before_predicting(image, fixed_size):
    """
    # process the image for predicting
    # read image by OpenCV
    # params return: image_original, image_processed, image_original_size
    # (original_image) shape -> (w, h, 3)
    # (image_processed) shape -> (1, 608, 608, 3)
    """

    image_original = image
    image_original_size = image.shape[:2]

    # original image shape (w, h, 3) -> predict image shape(1, 608, 608, 3)
    image_processed = utils.image_preprocess(np.copy(image_original), [fixed_size, fixed_size]) 

    return image_original, image_processed, image_original_size





# detecting info
# fromat： [class_name] -> [score] -> [coordinate]
def detecting_info(bboxes):
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    for bbox in bboxes:
        coordinate = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_index = int(bbox[5])
        class_name = CLASSES[class_index]
        score = '%.4f' % score
        xmin, ymin, xmax, ymax = list(map(str, coordinate))

        print(f"{class_name:8}: {score:6}, coordinate: ({xmin}, {ymin}, {xmax}, {ymax})")
    logging.info("Detecting done...")



# load video()
def video_load(video_path, video_output_path, output_format="XVID"):

      # load video
      vid = cv2.VideoCapture(video_path)
      logging.info("Video Loaded.")

      # by default VideoCapture returns float instead of int
      width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = int(vid.get(cv2.CAP_PROP_FPS))
      codec = cv2.VideoWriter_fourcc(*output_format)

      # save video
      video_saver = cv2.VideoWriter(video_output_path, codec, fps, (width, height))


      return vid, video_saver



# create yolov4 model directly
# return: model

def yolov4_model_create(input_size, 
                        num_class=len(utils.read_class_names(cfg.YOLO.CLASSES))
                        ):
    """
    # input layer 
    # YOLOv4 -> return [conv_sbbox, conv_mbbox, conv_lbbox]
    # return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes] contains (x, y, w, h, score, probability)
    # create model
    """

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    sml_feature_maps = YOLOv4(input_layer, num_class) # Return [conv_sbbox, conv_mbbox, conv_lbbox], len = 3
 
    output_layers = []
    # decode -> output_layers
    for feature_map in sml_feature_maps:
        # print("feature_map: ", feature_map)
        output_layer = decode(feature_map, num_class)
        # print("decode -> ",output_layer.shape)
        output_layers.append(output_layer)


    model = tf.keras.Model(input_layer, output_layers, name="YOLOv4_jam")      # create model
    # model.summary()
    logging.info("YOLOv4 Model built.")


    return model 


def yolov4_model_create_4_train(input_size=cfg.TRAIN.INPUT_SIZE, 
                        num_class=len(utils.read_class_names(cfg.YOLO.CLASSES)),
                        anchors=utils.get_anchors(cfg.YOLO.ANCHORS),
                        strides=np.array(cfg.YOLO.STRIDES),
                        xyscale=cfg.YOLO.XYSCALE
                        ):
    
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, num_class)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode_train(fm, num_class, strides, anchors, i, xyscale)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors, name="YOLOv4_4_training")
    logging.info("YOLOv4 Model built.")

    return model




# draw_bbox
# use OpenCV funcrions to draw
def draw_bbox(image, bboxes, classes=utils.read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    # colors
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # random color
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.6
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



# bboxes_optimizing!! 
# 两个 post process bbox + nms 
def bboxes_optimizing(bboxes_pred, image_original_size, input_size,
                      anchors=utils.get_anchors(cfg.YOLO.ANCHORS), 
                      strides=np.array(cfg.YOLO.STRIDES), 
                      xyscale=cfg.YOLO.XYSCALE, 
                      score_thr=cfg.YOLO.SCORE_THRESHOLD, iou_thr=cfg.YOLO.IOU_THRESHOLD, nms_method="nms",
                      show_info=False, show_detected_objects_numbers=True):
    # NMS to filter Bboxes
    # param bboxes: (xmin, ymin, xmax, ymax, score, class)
    pred_bbox = utils.postprocess_bbbox(bboxes_pred, anchors, strides, xyscale)
    bboxes = utils.postprocess_boxes(pred_bbox, image_original_size, input_size, score_threshold=score_thr)
    bboxes_processed = utils.nms(bboxes, iou_threshold=iou_thr, method=nms_method)

    # logging.info("Detecting finished.")
    if show_detected_objects_numbers:
      logging.info(f"Found {len(bboxes_processed)} objects.") 


    if show_info:
      detecting_info(bboxes_processed)

    return bboxes_processed




# save whole yolo model
def save_yolov4_model(input_size, output_path, weights=cfg.YOLO.WEIGHTS):

  model = yolov4_model_create(input_size=input_size)

  # load darknet weights
  utils.load_darknet_weights(model, weights)
  logging.info("weights loaded.") 
  # print(model.summary())
  # print("layers:", len(model.layers))

  model.save(output_path)
  print("saved!")



# load YOLOv4 model with weights
def load_yolov4_model(yolov4_model_path):
    return tf.keras.models.load_model(yolov4_model_path)
    





