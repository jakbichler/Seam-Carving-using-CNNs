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


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to the input image')
parser.add_argument('--class_id', type=int, help='Class to perform Grad-CAM with')
args = parser.parse_args()
image_path = args.image_path
class_id = args.class_id


# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        

        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # get the pretrained VGG19 network
        self.net = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.relevant_layer = self.net.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.net.classifier
        
        # placeholder for the gradients
        self.gradients = None

   

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.relevant_layer(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    

    def grad_cam_heatmap(self, class_id, img):

        # get the most likely prediction of the model
        pred_all = ic(self.forward(img))

        pred_index = ic(pred_all.argmax(dim=1))

        # get the gradient of the output with respect to the parameters of the model
        pred_all[:, class_id].backward()

        # pull the gradients out of the model
        gradients = self.get_activations_gradient()
        ic(gradients.shape)
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        ic(pooled_gradients.shape)
        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()
        ic(activations.shape)

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        ic(heatmap.shape    )
        # relu on top of the heatmap
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        return heatmap/torch.max(heatmap)

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.relevant_layer(x)
    
