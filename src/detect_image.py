import cv2 as cv
import yolo

data = 'testImage.jpg'

def main():
	# YOLO v3
	model = '../YOLO_COCO/'
	inputResolution = (416,416)
	detector = yolo.TrainedObjectDetector(model, inputResolution)

	frame = cv.imread(f'../data/{data}')
	detections = detector.predict(frame)
	detector.drawBoxes(frame, detections)

	cv.namedWindow('Live', cv.WINDOW_NORMAL)
	cv.imshow('Live', frame)
	cv.imwrite(f'../output/{data}', frame)
	cv.waitKey(0)

if __name__ == '__main__':
    main()