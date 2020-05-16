import cv2 as cv
import glob
import yolo
import ssd

def main():
	# data
	# dataPath = r'C:\Users\Mahmoud Dahmani\Downloads\object-detection-crowdai\*'
	dataPath = '../data/images/*'
	dataset = sorted(glob.glob(dataPath))

	# model
	model = '../YOLO_COCO/'
	# model = '../SSD_COCO/'
	detector = yolo.PreTrainedDetector(model)

	for image in dataset:
		frame = cv.imread(image)
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)
		# detector.logResults(frame, image[len(dataPath)-1:-4], detections)

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		cv.imwrite(f'../output/{image[len(dataPath)-1:]}', frame)
		if cv.waitKey(1) >= 0:
			break

if __name__ == '__main__':
    main()