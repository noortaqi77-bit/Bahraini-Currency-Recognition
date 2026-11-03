# Bahraini Currency Recognition
This project focuses on detecting and identifying Bahraini currency using YOLO26, a deep learning model for object detection.

The model recognizes different Bahraini currency denominations and locates them in images by drawing bounding boxes around detected objects.
#### App :
https://bahraini-currency.streamlit.app/


### Dataset
The dataset was collected by our team. It contains images of Bahraini banknotes and coins across 10 classes:

5 fils, 10 fils, 25 fils, 50 fils, 100 fils, 0.5 BD, 1 BD, 5 BD, 10 BD, and 20 BD.

### Model

YOLO26 was used to train an object detection model that predicts the currency class and location in an image.

### Evaluation Results

The model was evaluated on the test set using the following metrics:
Test Results:
- Precision       91.1%
- Recall          86.7%
- mAP@50          89.6%
- mAP@50-95       75.1%

These metrics measure the correctness of detections, the proportion of actual objects detected, and the model's localization performance at different IoU thresholds.
