import numpy as np
import cv2 as cv
from time import time

class PreTrainedDetector:
	def __init__(self, model):
		self.inputResolution = (300, 300)
		labelsPath = f'{model}coco.names'
		weightsPath = f'{model}MobileNetSSD_deploy.caffemodel'
		configPath = f'{model}MobileNetSSD_deploy.prototxt.txt'
		self.labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
	                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
	                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
	                    'sofa', 'train', 'tvmonitor'] # COCO class labels (20 classes)
		np.random.seed(1)
		self.classColors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
		
		# SSD
		self._loadModel(weightsPath, configPath) 

	def predict(self, frame, minimumConfidence=0.3):
		frameHeight, frameWidth = frame.shape[:2]
		preprocessedFrame = cv.dnn.blobFromImage(frame, 0.007843, self.inputResolution, 127.5)
		self.net.setInput(preprocessedFrame)

		start = time()
		outputs = self.net.forward()
		end = time()
		print(f'Forward pass of a single frame took {end-start:.3f} s')

		return self._decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence)

	def drawBoxes(self, frame, detections):
		indices, boxes, classIDs, confidences = detections
		if len(indices) > 0:
			for i in indices.flatten():
				x, y, w, h = boxes[i]
				caption = f'{self.labels[classIDs[i]]}: {confidences[i]:.3f}'
				classColor = self.classColors[classIDs[i]].tolist()

				cv.rectangle(frame, (x, y), (x+w, y+h), classColor, 2)
				cv.putText(frame, caption, (x, y-5), cv.LINE_AA, 0.5, classColor, 2)

	def _loadModel(self, weightsPath, configPath):
		self.net = cv.dnn.readNetFromCaffe(configPath, weightsPath) # pre-trained on COCO

	@staticmethod
	def _decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence):
		boxes, confidences, classIDs = [], [], []
		for i in np.arange(outputs.shape[2]):
			confidence = outputs[0, 0, i, 2]
			if confidence > minimumConfidence:
				classID = int(outputs[0, 0, i, 1]) 
				box = outputs[0, 0, i, 3:7] * [frameWidth, frameHeight, frameWidth, frameHeight]
				startX, startY, endX, endY = box.astype('int')
				width = endX - startX
				height = endY - startY
				boxes.append([startX, startY, width, height])
				confidences.append(confidence)
				classIDs.append(classID)

		indices = np.arange(len(boxes))

		return (indices, boxes, classIDs, confidences)

	def logResults(self, frame, frameName, detections):
		indices, boxes, classIDs, confidences = detections
		frameHeight, frameWidth = frame.shape[:2]

		with open(f'../output/{frameName}.txt', 'w') as file:
			if len(indices) > 0:
				for i in indices.flatten():
					x, y, w, h = boxes[i]
					left, top, right, bottom = x, y, x+w, y+h
					if left < 0:
						left = 0
					if right >= frameWidth:
						right = frameWidth - 1
					if top < 0:
						top = 0
					if bottom >= frameHeight:
						bottom = frameHeight - 1
					file.write(f'{self.labels[classIDs[i]].replace(' ','')} {confidences[i]} {left} {top} {right} {bottom}\n')