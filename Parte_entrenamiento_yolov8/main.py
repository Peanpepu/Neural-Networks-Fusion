from ultralytics import YOLO

# Load a model
#model = YOLO("/root/TFG/yolov8m.yaml")  # build a new model from scratch
# model = YOLO("model/best_rgb1.pt")  # load a pretrained model (recommended for training)
model = YOLO("model/yolov8m.pt")  # load a pretrained model (recommended for training)

# Add desired hyperparameters
hyp = dict()
hyp['lr0']= 0.01 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
hyp['lrf']= 0.002 # final learning rate (lr0 * lrf)
hyp['momentum']= 0.937 # SGD momentum/Adam beta1
hyp['weight_decay']= 0.0005 # optimizer weight decay 5e-4
hyp['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
hyp['warmup_momentum']= 0.8 # warmup initial momentum
hyp['warmup_bias_lr']= 0.1 # warmup initial bias lr
hyp['box']= 7.5 # box loss gain
hyp['cls']= 0.5 # cls loss gain (scale with pixels)
hyp['dfl']= 1.5 # dfl loss gain
hyp['pose']= 12.0 # pose loss gain
hyp['kobj']= 1.0 # keypoint obj loss gain
hyp['label_smoothing']= 0.0 # label smoothing (fraction)
hyp['nbs']= 64 # nominal batch size
hyp['hsv_h']= 0.015 # image HSV-Hue augmentation (fraction)
hyp['hsv_s']= 0.7 # image HSV-Saturation augmentation (fraction)
hyp['hsv_v']= 0.4 # image HSV-Value augmentation (fraction)
hyp['degrees']= 0.0 # image rotation (+/- deg)
hyp['translate']= 0.1 # image translation (+/- fraction)
hyp['scale']= 0.2 # image scale (+/- gain)
hyp['shear']= 0.2 # image shear (+/- deg)
hyp['perspective']= 0.0 # image perspective (+/- fraction), range 0-0.001
hyp['flipud']= 0.7 # image flip up-down (probability)
hyp['fliplr']= 0.2 # image flip left-right (probability)
hyp['mosaic']= 0.0 # image mosaic (probability)
hyp['mixup']= 0.0 # image mixup (probability)
hyp['copy_paste']= 0.0 # segment copy-paste (probability)

# Use the model
model.train(hyp=hyp, data="/root/TFG/custom_dataset.yaml", epochs=120)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format

