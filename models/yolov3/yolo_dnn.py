import cv2 as cv
import numpy as np
from icecream import ic
import os
import torch
import torch.nn as nn
from torchvision import models, transforms


### This code is based on this GitHub: https://github.com/MrBam44/YOLO-object-detection-using-Opencv-with-Python/tree/main

class yolov3:

    def __init__(self) -> None:

        # Get the directory where yolo.py is located
        script_dir = os.path.dirname(__file__)

        # Construct the full paths to YOLO files
        weights_path = os.path.join(script_dir, "yolov3.weights")
        cfg_path = os.path.join(script_dir, "yolov3.cfg")

        # Load YOLO network
        self.net = cv.dnn.readNet(weights_path, cfg_path)

        self.classes = []
        with open("class.names", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        # print(classes)
        self.layer_name = ic(self.net.getLayerNames())



        # Slice the layers until the last convolution (conv105)
        layers_until_conv105 = ic(self.layer_name[:(len(self.layer_name)-2)])
        layers_after_conv105 = ic(self.layer_name[(len(self.layer_name)-2):])

        
        
        # 3 output layers ['yolo_82', 'yolo_94', 'yolo_106']
        self.output_layers = ic([self.layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()])

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.gradients = None



    def activations_hook(self, grad):
        self.gradients = grad




    def forward(self, image_path):
        # Load Image
        img = cv.imread(image_path)
        img = cv.resize(img, None, fx=0.4, fy=0.4)
        height, width, channel = img.shape

        # Detect Objects
        blob = cv.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)




        # Outs is touple of 3 elements corresponding to the 3 heads of the YOLO network and then an array of size (..., 85)
        outs = self.net.forward(self.output_layers)
        # ic(outs[0].shape)


        last_conv_layer_name = ic(self.layer_name[-3])
        last_conv_layer = self.net.getLayer(last_conv_layer_name)


        out_conv = ic(self.net.forward(last_conv_layer_name).shape)

        # Register hook to the last conv layer
        hook = last_conv_layer.registerBackwardHook(self.activations_hook)




        # Forward pass
       



        # Showing Information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # print(len(boxes))
        # number_object_detection = len(boxes)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        ic(indexes.shape)
        #print(indexes)

        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                # print(label)
                color = self.colors[i]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + 30), font, 1, color, 1)

        cv.imshow("Image with detections, q to quit", img)
        key=cv.waitKey(0)
        if key == ord('q'):  # Press 'q' to exit
            cv.destroyAllWindows()

    


if __name__ == '__main__':
    # This code won't run if this file is imported.
    image_path = "../../data/images/cat_dog.jpg"

    model = yolov3()
    ic(model.output_layers)
    model.forward(image_path)
