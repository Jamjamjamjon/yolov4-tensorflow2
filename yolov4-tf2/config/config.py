
from easydict import EasyDict as edict
from datetime import datetime


# cfg =  __C
__C                           = edict() 
cfg                           = __C

# Global variable
__C.SAVE_OR_NOT               = True



# YOLO options
__C.YOLO                      = edict()


# For Detecting
__C.YOLO.INPUT_SIZE			  = 608
__C.YOLO.STRIDES              = [8, 16, 32]  
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]        
__C.YOLO.CLASSES              = "./config/classes/coco.names"




__C.YOLO.ANCHORS              = "./config/anchors/yolov4_anchors.txt"

__C.YOLO.ANCHOR_PER_SCALE     = 3       
__C.YOLO.IOU_LOSS_THRESH      = 0.5     


__C.YOLO.IOU_THRESHOLD       = 0.6
__C.YOLO.SCORE_THRESHOLD     = 0.213    

__C.YOLO.IMAGE_PATH_EXAMPLE     = "./data/images/kite.jpg"

__C.YOLO.WEIGHTS				="./weights/yolov4.weights"
__C.YOLO.MODEL_PATH                  = "./saved_model/yolov4_170036.h5"

__C.YOLO.IMAGE_OUTPUT_PATH      = "./data/outputs/result_" + datetime.now().strftime("%d%H%M%S") + ".jpg"
__C.YOLO.MODEL_OUTPUT_PATH      = "./saved_model/yolov4_" + datetime.now().strftime("%d%H%M") + ".h5"
__C.YOLO.VIDEO_OUTPUT_PATH      = "./data/outputs/result_" + datetime.now().strftime("%d%H%M%S") + ".mp4"





# Train 
# Train 
# Train 

__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/mnist/mnist_train.txt"
__C.TRAIN.BATCH_SIZE          = 2
__C.TRAIN.INPUT_SIZE          = 608         # one of those [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG            = True        
__C.TRAIN.LR_INIT             = 1e-3        
__C.TRAIN.LR_END              = 1e-6        
__C.TRAIN.WARMUP_EPOCHS       = 5           
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20        
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30        



# EVALUATE 

__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/coco/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 608
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/coco/coco_val_detected/"
__C.TEST.SCORE_THRESHOLD      = 0.6
__C.TEST.IOU_THRESHOLD        = 0.213


