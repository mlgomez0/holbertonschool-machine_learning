#!/usr/bin/env python3
"""class FaceAlign"""
import dlib


class FaceAlign():
    """align faces"""
    def __init__(self, shape_predictor_path):
        """initializes a FaceAlign instance"""
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
