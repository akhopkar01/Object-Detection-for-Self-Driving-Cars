import cv2 as cv
import glob
import yolo

def main():
	# data
	dataPath = '../data/images/*'
	dataset = sorted(glob.glob(dataPath))

	# model
	model = '../YOLO_COCO/'
	inputResolution = (416,416)
	detector = yolo.TrainedObjectDetector(model, inputResolution)

	for image in dataset:
		frame = cv.imread(image)
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)
		detector.logResults(image[len(dataPath)-1:-4], detections)

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		cv.imwrite(f'../output/{image[len(dataPath)-1:]}', frame)
		if cv.waitKey(1) >= 0:
			break

if __name__ == '__main__':
    main()