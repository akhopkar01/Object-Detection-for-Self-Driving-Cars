import numpy as np
import cv2 as cv
from time import time

class TrainedObjectDetector:
	def __init__(self, model, inputResolution):
		labelsPath = f'{model}coco.names'
		weightsPath = f'{model}yolov3.weights'
		configPath = f'{model}yolov3.cfg'

		self.labels = open(labelsPath).read().strip().split('\n') # COCO class labels (80 classes)
		np.random.seed(1)
		self.classColors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
		
		# YOLO v3
		self._loadModel(weightsPath, configPath) 
		self.inputResolution = inputResolution

	def predict(self, frame, minimumConfidence=0.5, minimumOverlap=0.3):
		frameHeight, frameWidth = frame.shape[:2]
		preprocessedFrame = cv.dnn.blobFromImage(frame, 1/255, self.inputResolution, swapRB=True, crop=False)
		self.net.setInput(preprocessedFrame)

		start = time()
		outputs = self.net.forward(self.outputLayers)
		end = time()
		print(f'Forward pass of a single frame took {end-start:.3f} s')

		return self._decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence, minimumOverlap)

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
		self.net = cv.dnn.readNetFromDarknet(configPath, weightsPath) # pre-trained on COCO
		layers = self.net.getLayerNames()
		self.outputLayers = [layers[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

	@staticmethod
	def _decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence, minimumOverlap):
		boxes, confidences, classIDs  = [], [], []
		for output in outputs:
			for detection in output:
				scores = detection[5:] # class probabilities
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions 
				if confidence > minimumConfidence:
					# map box parameters from YOLO representation to opencv representation
					box = detection[0:4] * [frameWidth, frameHeight, frameWidth, frameHeight]
					centerX, centerY, width, height = box
					x = centerX - width/2
					y = centerY - height/2
				
					boxes.append([int(x), int(y), int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maximum suppression
		indices = cv.dnn.NMSBoxes(boxes, confidences, minimumConfidence, minimumOverlap)

		return (indices, boxes, classIDs, confidences)

	def logResults(self, frameName, detections):
		indices, boxes, classIDs, confidences = detections
		with open(f'../output/{frameName}.txt', 'w') as file:
			if len(indices) > 0:
				for i in indices.flatten():
					x, y, w, h = boxes[i]
					left, top, right, bottom = x, y, x+w, y+h
					file.write(f"{self.labels[classIDs[i]].replace(' ','')} {confidences[i]} {left} {top} {right} {bottom}\n")