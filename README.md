# Single-Stage Detections Algorithms        			             
This project implements YOLO-based and SSD-based object detectors for self-driving cars.

<p align="center">
  <img src="https://github.com/dahhmani/Object-Detection-for-Self-Driving-Cars/blob/master/Inference/output/testImage.jpg?raw=true">
</p>

# Setup
## Inference
1. Navigate to Inference directory
2. Put the your dataset in the corresponding data folder
3. Download the pretrained model weights and put them in COCO-YOLO/COCO-SSD
4. Navigate to src directory
5. run ```python detect_image.py```/```python detect_video.py``` in the terminal (for detection on images/videos)
  
## Evaluation
1. Navigate to Evaluation directory
2. Put the ground truth and predicted results in the input folder
3. run ```python evaluateDetector.py``` in the terminal


