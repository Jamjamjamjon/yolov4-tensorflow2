from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import yolo_model.utils as utils
from config.config import cfg
from yolo_model.yolov4 import YOLOv4
from yolo_model.utils import decode
from datetime import datetime
import yolo_model.api as api






flags.DEFINE_string("weights", cfg.YOLO.WEIGHTS, "path to weights file")
flags.DEFINE_integer("input_size", 608, "resize images to")
flags.DEFINE_string("annotation_path", "./data/coco/val2017.txt", "annotation path")
flags.DEFINE_string("write_image_path", cfg.TEST.DECTECTED_IMAGE_PATH, "write detected image path")



def main(_argv):

    # file ops
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)



    # GPU or CPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU!!!!!!!!!!!")
    else:
        print("NO GPU!!!!!!!!!!!")



    # setting prep
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)


    # create model
    model = api.yolov4_model_create(FLAGS.input_size)
    # laod weights
    api.load_darknet_weights(model, cfg.YOLO.WEIGHTS)




       
     
    num_lines = sum(1 for line in open(cfg.TEST.ANNOT_PATH))  # get the number of line of annotation
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        # example annotation
        # ./data/coco/images/val2017/000000289343.jpg 473,395,511,
        #                                             423,16 204,235,264,
        #                                             412,0 0,499,339,605,
        #                                             13 204,304,256,456,1
        for num, line in enumerate(annotation_file):
            # strip()删除开头和结尾的\t\r\n这些字符
            # split()返回各块的一个列表
            # 以上例子返回如下:
            # ['./data/coco/images/val2017/000000289343.jpg', 
            #  '473,395,511,423,16',
            #  '204,235,264,412,0',
            #  '0,499,339,605,13', 
            #  '204,304,256,456,1']

            annotation = line.strip().split()       
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # get ground truth bbox data -> [x1, y1, x2, y2, class]
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)


            """
            ground-truth：指的是真实框的txt
            对应绘制mAP的 get_gt_txt.py 文件
            结果为：
            3 194 400 208 414
            0 43 372 57 386
            0 277 201 291 215
            1 143 134 199 190
            1 299 49 341 91
            5 150 218 192 260
            5 303 170 331 198
            7 101 92 129 120
            1 150 293 206 349
            5 0 102 112 214
            8 200 89 312 201
            """
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')



            
            image_original, image_processed, image_original_size = api.image_preprocess_before_predicting(image=image, 
                                                                                                  fixed_size=FLAGS.input_size)
            # predict
            # image shape -> (1, 608, 608, 3)
            bboxes_predicted = model.predict(image_processed)
            bboxes_processed = api.bboxes_optimizing(bboxes_pred=bboxes_predicted, 
                                                         image_original_size=image_original_size, 
                                                         input_size=FLAGS.input_size, 
                                                         show_info=False,
                                                         show_detected_objects_numbers=False)

            # Predict Process
            # image_size = image.shape[:2]
            # image_data = utils.image_preporcess(np.copy(image), [input_size, input_size])

            # pred_bbox = model.predict(image_data)
            
            # XYSCALE = cfg.YOLO.XYSCALE
            # pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=XYSCALE)

            # pred_bbox = tf.concat(pred_bbox, axis=0)
            # bboxes = utils.postprocess_boxes(pred_bbox, image_size, input_size, cfg.TEST.SCORE_THRESHOLD)
            # bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


            # save predicted test images
            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image = api.draw_bbox(image, bboxes_processed)
                cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)

    
            """
            detection-results：指的是预测结果的txt
            对应绘制mAP的 get_dr_txt.py文件
            结果为：
            0 0.9426 277 201 291 214
            0 0.9347 43 372 57 386
            1 0.9877 143 133 199 189
            1 0.9842 150 293 205 348
            1 0.9663 299 49 341 90
            5 0.9919 302 169 330 198
            5 0.9823 0 102 112 213
            5 0.9684 150 218 190 259
            7 0.9927 101 92 129 119
            8 0.9695 199 88 314 202

            """
            with open(predict_result_path, 'w') as f:
                for bbox in bboxes_processed:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


