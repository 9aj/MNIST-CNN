# MNIST-CNN
Experimenting with TensorFlow 2.0 for Apple Silicon. A simple 3 layer ConvNet architecture was implemented, with 2 maxpool layers with kernel size 2x2. This project was completed as a brief introduction into CNN construction for later projects.

### Model Accuracy and Loss
Loss: 2.037%
Model Accuracy: 99.290%

### Architecture Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                10250     
=================================================================
Total params: 66,570
Trainable params: 66,570
Non-trainable params: 0
```
