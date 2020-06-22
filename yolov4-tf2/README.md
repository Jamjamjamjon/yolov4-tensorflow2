Minimal yolov4 with tensorflow2



# usage 

## image detect
python yolo_demo.py
python yolo_demo.py --image_path [image_path] 

## video detect
python yolo_demo.py --video_path [video_path] 


# ！！！
train_with_gradientTape.py won't work well!
Try not use it!



# not yet

- train.py file
- NO keras.model.fit()



## done
- tf2
- load darknet weights 
- YOLOv4 module(CSP_darknet53 + SPP + modified PAN)
- Mish
- image preprocess
- absl_python
- diou_nms
- ciou loss
- Eager mode training with `tf.GradientTape`
- model.evaluate（）
- api.py
- yolo class
- detect video demo (yolo class)
- detect image demo (yolo class)
- image_detect_demo.py
- evaluate on coco dataset



# some data shape info

1. input_layer -> [608, 608, 3] , keras.Input()
2. YOLOv4 -> sbox: (none, 76, 76, 255),
                -> mbox: (none, 38, 38, 255),
                -> lbox: (none, 19, 19, 255)

3. feed to decode, outputs_layer -> sbox: (none, none, none, 3, 85),
                  -> mbox: (none, none, none, 3, 85),
                  -> lbox: (none, none, none, 3, 85)
    85 -> (x, y, w, h, confidence, num_class) = (4 + 1 + 80)

4. YOLOv4 model -> keras.model(input_layer, outputs_layer)
5.model.predict() -> (1, 608, 608, 3)














