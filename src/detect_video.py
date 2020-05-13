import cv2 as cv
import yolo

video = 'testVideo.mp4'

def main():
	# data
	inputVideo = cv.VideoCapture(f'../data/video/{video}')
	outputVideo = cv.VideoWriter(f'../output/{video}', cv.VideoWriter_fourcc(*'XVID'), 30, (int(inputVideo.get(3)),int(inputVideo.get(4))))
	
	# model
	model = '../YOLO_COCO/'
	inputResolution = (416,416)
	detector = yolo.TrainedObjectDetector(model, inputResolution)

	frameIndex = 0
	while True:
		frameIndex += 1
		read, frame = inputVideo.read()
		if not read:
			break	
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)
		detector.logResults(f'{video[:-4]}_{frameIndex}', detections)

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		outputVideo.write(frame)
		if cv.waitKey(1) >= 0:
			break

if __name__ == '__main__':
    main()