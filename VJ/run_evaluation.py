import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import seaborn as sns
import matplotlib.pylab as plt

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self, neighbours, scale, alpha, beta):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        import detectors.your_super_detector.ear_detector as super_detector
        cascade_detector = super_detector.Detector()
        
        predictions = []
        annotations = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)
            # Apply some preprocessing
            #img = cv2.fastNlMeansDenoising(img, None)
            #img = preprocess.blur(img, filter, kernel)
            #img = preprocess.resize(img, resize)
            #img = preprocess.histogram_equalization_rgb(img) # This one makes VJ worse
            img = preprocess.change_contrast_brightness(img, alpha, beta)
            #img = preprocess.edge_enhancment(img)
            
            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img, neighbours, scale)
            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)
            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)
            predictions.append(prediction_list)
            annotations.append(annot_list)

        miou = np.average(iou_arr)
        return miou

if __name__ == '__main__':
    ev = EvaluateAll()
    neighbours = 2
    scale_factor = 1.015
    alpha = 0.75
    beta = -10
    miou = ev.run_evaluation(neighbours, scale_factor, alpha, beta)
    print("Average IOU:", f"{miou:.2%}")
    print("\n")
    '''
    for i in range(1,3,1):
        results = []
        labels_x = []
        for j in range(0,100,10):
            miou = ev.run_evaluation(2,1.015,i,j)
            results.append(miou)
            labels_x.append('%.2f%%' % (miou*100))
        param_matrix.append(results)
        labels_y.append(labels_x)
    print(param_matrix)
    print(labels_y)
    #x_axis_labels = kernels # labels for x-axis
    #y_axis_labels = filters # labels for y-axis
    ax = sns.heatmap(param_matrix, annot=labels_y, fmt = '')
    ax.set_ylabel('Alpha')
    ax.set_xlabel('Beta')
    plt.show()'''
