#!/usr/bin/env python3
"""class for model Yolo"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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

    def sigmoid(self, z):
        """performs sigmoid mapping"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """Returns a tuple of (boxes, box_confidences,
           box_class_probs)"""
        boxes = []
        confidences = []
        class_proba = []
        img_H = image_size[0]
        img_W = image_size[1]
        for output in outputs:
            boxes.append(output[..., 0:4])
            confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            class_proba.append(self.sigmoid(output[..., 5:]))
        for i, box in enumerate(boxes):
            g_h, g_w, achors_box, _ = box.shape
            coordidate = np.zeros((g_h, g_w, achors_box))
            idx_y = np.arange(g_h)
            idx_y = idx_y.reshape(g_h, 1, 1)
            idx_x = np.arange(g_w)
            idx_x = idx_x.reshape(1, g_w, 1)
            C_x = coordidate + idx_x
            C_y = coordidate + idx_y
            centerX = box[..., 0]
            centerY = box[..., 1]
            width = box[..., 2]
            height = box[..., 3]
            bx = (self.sigmoid(centerX) + C_x) / g_w
            by = (self.sigmoid(centerY) + C_y) / g_h
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (np.exp(width) * pw) / self.model.input.shape[1].value
            bh = (np.exp(height) * ph) / self.model.input.shape[2].value
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            box[..., 0] = x1 * img_W
            box[..., 1] = y1 * img_H
            box[..., 2] = x2 * img_W
            box[..., 3] = y2 * img_H
        return boxes, confidences, class_proba

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Returns a tuple of (filtered_boxes, box_classes,
           box_scores)"""
        filtered_scores = []
        filtered_class = []
        filtered_boxes = []
        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            box_max_scores = np.max(box_score, axis=-1).reshape(-1)
            box_class_idx_del = np.where(box_max_scores < self.class_t)
            mask = box_max_scores >= self.class_t
            score = mask * box_max_scores
            score1 = score[score > 0]
            filtered_scores.append(score1)
            class1 = np.argmax(box_score, axis=-1).reshape(-1)
            class2 = np.delete(class1, box_class_idx_del)
            filtered_class.append(class2)
            a, b, c, _ = boxes[i].shape
            mask_reshape = mask.reshape(a, b, c, 1)
            box1 = boxes[i] * mask_reshape
            box2 = box1[box1 != 0]
            filtered_boxes.append(box2)
        filtered_scores1 = np.concatenate(filtered_scores)
        filtered_class1 = np.concatenate(filtered_class)
        filtered_boxes1 = np.concatenate(filtered_boxes)
        filtered_boxes2 = filtered_boxes1.reshape(-1, 4)
        return filtered_boxes2, filtered_class1, filtered_scores1

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Returns a tuple of
           (box_predictions, predicted_box_classes,
            predicted_box_scores)"""
        index = np.lexsort((-box_scores, box_classes))
        box_predictions = filtered_boxes[index]
        predict_box_classes = box_classes[index]
        predict_box_scores = box_scores[index]
        _, number_counts = np.lib.arraysetops.unique(predict_box_classes,
                                                     return_counts=True)
        i = 0
        acummulated = 0
        for number_count in number_counts:
            while i < acummulated + number_count:
                j = i + 1
                while j < acummulated + number_count:
                    b_p1 = box_predictions[i]
                    b_p2 = box_predictions[j]
                    x_x1 = np.maximum(b_p1[0], b_p2[0])
                    y_y1 = np.maximum(b_p1[1], b_p2[1])
                    x_x2 = np.minimum(b_p1[2], b_p2[2])
                    y_y2 = np.minimum(b_p1[3], b_p2[3])
                    inter_area = (y_y2 - y_y1) * (x_x2 - x_x1)
                    box1_area = (b_p1[3] - b_p1[1])*(b_p1[2] - b_p1[0])
                    box2_area = (b_p2[3] - b_p2[1])*(b_p2[2] - b_p2[0])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area/union_area
                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions,
                                                    j, axis=0)
                        predict_box_scores = np.delete(predict_box_scores,
                                                       j, axis=0)
                        predict_box_classes = np.delete(predict_box_classes,
                                                        j, axis=0)
                        number_count -= 1
                    else:
                        j += 1
                i += 1
            acummulated += number_count
        return box_predictions, predict_box_classes, predict_box_scores

    @staticmethod
    def load_images(folder_path):
        """Returns a tuple of (images, image_paths)
        """
        image_paths = glob.glob(folder_path + "/*")
        list_path = [cv2.imread(i) for i in image_paths]
        return list_path, image_paths

    def preprocess_images(self, images):
        """method to preprocess images for Darknet model"""
        images_list = []
        images_shape = []
        for img in images:
            images_shape.append([img.shape[0], img.shape[1]])
            new_size = (self.model.input.shape[1], self.model.input.shape[2])
            img_resized = (cv2.resize(img, new_size,
                           interpolation=cv2.INTER_CUBIC)) / 255
            images_list.append(img_resized)
        return (np.array(images_list), np.array(images_shape))

    def show_boxes(self, image, boxes,
                   box_classes, box_scores,
                   file_name):
        """Displays the image with all
           boundary boxes, class names,
           and box scores"""
        for i in range(len(boxes)):
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            w = int(boxes[i][2])
            h = int(boxes[i][3])
            score = str(round(box_scores[i], 2))
            label = self.class_names[box_classes[i]] + " " + score
            color = (255, 0, 0)
            color1 = (0, 0, 255)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
            cv2.putText(image, label, (x-5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color1, 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if not os.path.exists('detections'):
            os.mkdir("detections")
        path = "detections"
        if key == ord('s'):
            cv2.imwrite(os.path.join(path, file_name), image)
        cv2.destroyAllWindows()
