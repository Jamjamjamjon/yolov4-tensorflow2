
import yolo_model.utils as utils
from yolo_model.utils import decode, decode_train, freeze_all, unfreeze_all
from yolo_model.yolov4 import YOLOv4
from yolo_model.dataset import Dataset
from yolo_model.loss import compute_loss
import yolo_model.api as api
from config.config import cfg

import time
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS


# create——model-train


def main(_argv):


    # train data
    dataset_train = Dataset(dataset_type="train")

    print(type(dataset_train))
    
    # log_dir
    logdir_train = "logs/gradient_tape/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/train"

    steps_per_epoch = len(dataset_train)

    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS       # 20
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS     # 30
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch        #  5 * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES         = np.array(cfg.YOLO.STRIDES)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)


    LOAD_WEIGHTS = None
    
    model = api.yolov4_model_create_4_train(input_size=cfg.TRAIN.INPUT_SIZE)

    if LOAD_WEIGHTS == None:
        print("Training from scratch")
    else:
        utils.load_weights_tiny(model, FLAGS.weights)

    optimizer = tf.keras.optimizers.Adam()
    train_summary_writer = tf.summary.create_file_writer(logdir_train)


    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]        # 0,1    2,3   4,5
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = ciou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   ciou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               ciou_loss, conf_loss,
                                                               prob_loss, total_loss))
            
            # update learning rate 
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT     # cfg.TRAIN.LR_INIT = 1e-3,  cfg.TRAIN.LR_END =  1e-6
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())


            # writing summary data
            with train_summary_writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/ciou_loss", ciou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            train_summary_writer.flush()





    for epoch in range(first_stage_epochs + second_stage_epochs):

        for image_data, target in dataset_train:
            train_step(image_data, target)
            

        model.save_weights("./checkpoints/yolov4")







if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass