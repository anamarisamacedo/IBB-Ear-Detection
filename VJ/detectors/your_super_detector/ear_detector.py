import cv2, sys, os
import numpy as np

class Detector:
	# Ear detector
	cascade_left_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
	cascade_right_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	if cascade_left_ear.empty():
		raise IOError('Unable to load the left ear cascade classifier xml file')
	if cascade_right_ear.empty():
		raise IOError('Unable to load the right ear cascade classifier xml file')
		
	def detect(self, img, neighbours, scale):
		left_ear = self.cascade_left_ear.detectMultiScale(img, scale, neighbours)
		right_ear = self.cascade_right_ear.detectMultiScale(img, scale, neighbours)
		if len(right_ear) == 0:
			return left_ear
		elif len(left_ear) == 0:
			return right_ear
		else:
			ears = np.concatenate((left_ear, right_ear), axis=0)
			return tuple(map(tuple, ears))

'''
if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detector = Detector()
	detected_loc = detector.detect(gray)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imshow("Detection", img)
	cv2.waitKey(0) 
	cv2.destroyAllWindows() 
'''
