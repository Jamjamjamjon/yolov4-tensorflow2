import yolo_model.utils as utils
from yolo_model.utils import decode
from yolo_model.yolov4 import YOLOv4
import yolo_model.api as api
from config.config import cfg

import time
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS


# absl variables
flags.DEFINE_integer("input_size", 608, "input size of image")
flags.DEFINE_string("image_path", "./data/images/111.jpg", "path to input image")
flags.DEFINE_string("video_path", None, "path to input video")




# yolo class
class YOLO(object):

    # basic settings
    _default_settings = {
        "STRIDES": np.array(cfg.YOLO.STRIDES),
        "ANCHORS": utils.get_anchors(cfg.YOLO.ANCHORS),
        "SCORE_THRESHOLD": cfg.YOLO.SCORE_THRESHOLD,
        "IOU_THRESHOLD": cfg.YOLO.IOU_THRESHOLD,
        "NUM_CLASS": len(utils.read_class_names(cfg.YOLO.CLASSES)),  
        "CLASSES": utils.read_class_names(cfg.YOLO.CLASSES),
        "XYSCALE": cfg.YOLO.XYSCALE,
        "YOLOv4_WEIGHTS": cfg.YOLO.WEIGHTS,
        "IMAGE_OUTPUT_PATH": cfg.YOLO.IMAGE_OUTPUT_PATH,
        "VIDEO_OUTPUT_PATH": cfg.YOLO.VIDEO_OUTPUT_PATH,
        "SAVE_OR_NOT": cfg.SAVE_OR_NOT 
    }

    # init
    def __init__(self, input_size, **params):
        self.INPUT_SIZE = input_size
        self.__dict__.update(self._default_settings)
        

    # create model and load weights
    def create_model(self, input_size=cfg.YOLO.INPUT_SIZE,
                     darknet_weights=None
                     ):

        # create model
        model = api.yolov4_model_create(input_size)

        # laod weights
        if darknet_weights != None:
            api.load_darknet_weights(model, darknet_weights)

        return model


    # image detect
    def image_detecting(self, image, show_info=False):

        # process image
        # image must be opencv read
        image_original, image_processed, image_original_size = api.image_preprocess_before_predicting(image=image, 
                                                                                                      fixed_size=self.INPUT_SIZE)
    

        # create model
        # model = api.yolov4_model_create(self.INPUT_SIZE)
        # # laod weights
        # utils.load_darknet_weights(model, self.YOLOv4_WEIGHTS)

        # create model & laod weights
        model = self.create_model(self.INPUT_SIZE, self.YOLOv4_WEIGHTS)


        time_begin = time.time()

        # predict
        bboxes_predicted = model.predict(image_processed)
        # bbox process
        bboxes_processed = api.bboxes_optimizing(bboxes_pred=bboxes_predicted, 
                                                 image_original_size=image_original_size, 
                                                 input_size=self.INPUT_SIZE,
                                                 anchors=self.ANCHORS, strides=self.STRIDES, xyscale=self.XYSCALE, 
                                                 score_thr=self.SCORE_THRESHOLD, iou_thr=self.IOU_THRESHOLD, nms_method = "nms", 
                                                 show_info=show_info)

        # draw bbox
        image_with_bboxes = api.draw_bbox(image_original, bboxes_processed)

        
        time_end = time.time()
        logging.info(f"Time consumed: {time_end - time_begin}")


        # show
        image = Image.fromarray(image_with_bboxes)
        image.show()

        if self.SAVE_OR_NOT:
            image.save(self.IMAGE_OUTPUT_PATH)





    # video detect
    def video_detecting(self, video, video_saver, show_info=False):


        # create model
        # model = api.yolov4_model_create(self.INPUT_SIZE)

        # # laod weights
        # utils.load_darknet_weights(model, self.YOLOv4_WEIGHTS)

        # create model & laod weights
        model = self.create_model(self.INPUT_SIZE, self.YOLOv4_WEIGHTS)

        # time span list
        times = []
        time_begin = time.time()


        while True:
            """
                returns: ret,frame。
                - ret: bool. read correctly->true, end of the video -> false
                - frame: image, 3-dim
            """
            return_value, frame = video.read()

            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

            else:
                time_end = time.time()
                logging.info("Video detecting done.")
                logging.info(f"Time consumed: {time_end - time_begin} s")

                break


            frame, image_processed, frame_size = api.image_preprocess_before_predicting(image=frame, fixed_size=self.INPUT_SIZE)

            # # get size
            # frame_size = frame.shape[:2]

            # # frame(image) pre-porcess
            # image_processed = utils.image_preporcess(np.copy(frame), [self.INPUT_SIZE, self.INPUT_SIZE])

            # time which start to detect the frame
            prev_time = time.time()     
            # logging.info("Video detecting...")

            # predict
            bboxes_predicted = model.predict(image_processed)
            # bbox process
            bboxes_processed = api.bboxes_optimizing(bboxes_pred=bboxes_predicted, 
                                                     image_original_size=frame_size,
                                                     input_size=self.INPUT_SIZE, 
                                                     anchors=self.ANCHORS, strides=self.STRIDES, xyscale=self.XYSCALE, 
                                                     score_thr=0.25, iou_thr=0.213, nms_method = "nms", 
                                                     show_info=show_info)

            curr_time = time.time()     # time which start to detect the frame
            exec_time = curr_time - prev_time #  execute time.
            times.append(exec_time)     # save time span

            image = api.draw_bbox(frame, bboxes_processed)
            image = cv2.putText(image, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), 
                                (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if self.SAVE_OR_NOT:
                video_saver.write(image)
            cv2.imshow("result", result)


            if cv2.waitKey(1) & 0xFF == ord('q'): break




def main(argv):

    # GPU or CPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)



    #  YOLOv4 instance
    yolo = YOLO(input_size=FLAGS.input_size)

    # create model
    # yolo_model = yolo.create_model(input_size=FLAGS.input_size)
    # yolo_model.summary()


    if FLAGS.video_path == None:
        image = cv2.imread(FLAGS.image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo.image_detecting(image=image, show_info=False)

    else:
        video, video_saver = api.video_load(video_path=FLAGS.video_path, video_output_path = cfg.YOLO.VIDEO_OUTPUT_PATH) 
        yolo.video_detecting(video=video, video_saver=video_saver)


    




# ----------------------------

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



