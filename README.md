# VRDL_HW3

code for Selected Topics in Visual Recognition using Deep Learning Homework 3

## Hardware

- Ubuntu 16.04 LTS
- NVIDIA 1080ti

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in requirements.txt.

## Installation

1. Clone this repository
2. Install dependencies
```bach
pip3 install -r requirements.txt
```
3. Run setup from the repository root directory
```bach
python3 setup.py install
```
4. Install pycocotools from one of these repos.
- Linux: https://github.com/waleedka/coco

## Training
```bash
# Train a new model starting from ImageNet weights.  
# We use HW3's train_images as our dataset.  
python3 samples/coco/coco.py train --dataset=/path/to/train_images/ --model=imagenet

# Continue training the last model you trained. This will find the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/train_images/ --model=last
```

In coco.py, we preprocess the data and train our model.  
The model's weights will be saved every single epoch in logs/

## Output test.json by running test.py

```bach
python test.py
```

For testing data, we first use simliar dataloader to load testing images.  
Then we import the saved model.
```python
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs/')
model_path = model.find_last()
model.load_weights(model_path, by_name=True)
```
  
In the end, write 'image_id', 'category_id', 'segmentation' and 'score' result to a .josn file.  
Upload google drive submission.  
Best accuracy : mAP : 0.47644.  
