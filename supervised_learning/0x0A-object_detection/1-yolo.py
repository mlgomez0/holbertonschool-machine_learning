#!/usr/bin/env python3
"""class for model Yolo"""
import tensorflow.keras as K
import numpy as np
import tensorflow as tf


class Yolo():
    """class Yolo that uses Yolov3 algorithm"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """constructor for a Yolo class"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            lines = [line.split("\n")[0] for line in classes.readlines()]
        self.class_names = lines
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Returns a tuple of (boxes, box_confidences,
           box_class_probs)"""
        boxes = []
        confidences = []
        class_proba = []
        
        for output in outputs:
            scores = 1 / (1 + np.exp(-1 * output[:, :, :, 5:]))
            confidence = 1 / (1 + np.exp(-1 * output[:, :, :, 4:5]))
            box = output[:, :, :, 0:4] * np.concatenate((np.flipud(image_size), np.flipud(image_size)), axis=0)
            boxes.append(box)
            confidences.append(confidence)
            class_proba.append(scores)
        return boxes, confidences, class_proba
