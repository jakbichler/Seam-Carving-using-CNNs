import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import cv2
import argparse
from PIL import Image
from models.vgg19.vgg import CustomVGG
from utils import overlay_heatmap, modify_features

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--class_id', type=int, help='Class to perform Grad-CAM with')
args = parser.parse_args()
image_path = args.image_path
class_id = args.class_id


    
if __name__ == '__main__':

    # Initialize the CustomVGG model
    model = CustomVGG()
    model.eval()

    # Load an image
    img = Image.open(image_path)

    # save the shape of the image
    orig_img_shape = img.size


    # Transform and preprocess the image
    img = torch.FloatTensor(model.transform(img).unsqueeze(0))

    # Generate the Grad-CAM heatmap
    heatmap = model.grad_cam_heatmap(class_id, img)

    # # Display the heatmap using matplotlib
    # plt.matshow(heatmap.squeeze())
    # plt.show()


    # # Overlay the heatmap on the original image
    # overlayed_image = overlay_heatmap(heatmap, image_path)

    # # Convert the overlayed image to the appropriate depth
    # overlayed_image = cv2.convertScaleAbs(overlayed_image)
    
    # # Display the overlayed image using matplotlib
    # plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    # plt.show()

    inpainted_heatmap = modify_features(img, heatmap.squeeze(), image_path, orig_img_shape)
