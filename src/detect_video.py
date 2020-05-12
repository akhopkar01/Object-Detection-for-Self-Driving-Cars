import cv2 as cv
import yolo

data = 'testVideo.mp4'

def main():
	inputVideo = cv.VideoCapture(f'../data/{data}')
	outputVideo = cv.VideoWriter(f'../output/{data}', cv.VideoWriter_fourcc(*'XVID'), 30, (int(inputVideo.get(3)),int(inputVideo.get(4))))
	
	# YOLO v3
	model = '../YOLO_COCO/'
	inputResolution = (416,416)
	detector = yolo.TrainedObjectDetector(model, inputResolution)

	while True:
		read, frame = inputVideo.read()
		if not read:
			break	
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		outputVideo.write(frame)
		if cv.waitKey(1) >= 0:
			break

if __name__ == '__main__':
    main()