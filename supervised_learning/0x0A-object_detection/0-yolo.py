#!/usr/bin/env python3
import tensorflow.keras as K
import numpy as np
import glob

class Yolo():
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            lines = [line.split("\n")[0] for line in classes.readlines()]
        self.class_names = lines
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
