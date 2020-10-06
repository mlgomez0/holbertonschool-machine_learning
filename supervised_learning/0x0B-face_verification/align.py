#!/usr/bin/env python3
"""class FaceAlign"""
import dlib


class FaceAlign():
    """align faces"""
    def __init__(self, shape_predictor_path):
        """initializes a FaceAlign instance"""
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """Returns: a dlib.rectangle containing
           the boundary box for the face in the
           image, or None on failure"""
        boxes = self.detector(image)
        if len(boxes) >= 1:
            return max(boxes, key=lambda rect: rect.width() * rect.height())
        else:
            return dlib.rectangle(0, 0, image.shape[1], image.shape[0]) 
