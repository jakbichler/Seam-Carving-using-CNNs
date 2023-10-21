import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from icecream import ic
import cv2
import argparse
from PIL import Image
from models.vgg19.vgg import CustomVGG
from models.midas_depth.model_midas import MidasDepthEstimator
from utils import *
import os


    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--class_id', type=int, help='Class to perform Grad-CAM with')
    parser.add_argument('--n_seams', type=int, help='Number of seams to remove')
    parser.add_argument('--create_video', action='store_true', help='Whether to create a video or not')
    parser.add_argument('--depth_weight', type=float, help='Weight for the depth estimate', default=0.5)
    args = parser.parse_args()

    image_path = args.image_path
    class_id = args.class_id
    n_seams = args.n_seams
    create_video = args.create_video
    weight_depth = args.depth_weight

    os.system('clear')

    # Initialize the CustomVGG model
    print("\nInitializing the CustomVGG model...")
    obect_detector = CustomVGG()
    obect_detector.eval()


    # Initialize the MidasDepthEstimator model
    print("\nInitializing the MidasDepthEstimator model...")
    depth_estimator = MidasDepthEstimator()


    # Load the image
    orig_image = Image.open(image_path)
    orig_img_cv2 = cv2.imread(image_path)
    orig_img_shape = orig_image.size
    # Transform and preprocess the image
    img = torch.FloatTensor(obect_detector.transform(orig_image).unsqueeze(0))

    # Generate the Grad-CAM heatmap
    heatmap = obect_detector.grad_cam_heatmap(class_id, img)
    # Modify the feature map by painting on it (value range is 0-1)
    inpainted_heatmap = modify_features(img, heatmap.squeeze(), image_path, orig_img_shape)
    # Resize the heatmap to the original image size
    inpainted_heatmap = F.interpolate(inpainted_heatmap.unsqueeze(0).unsqueeze(0), size=(orig_img_shape[1], orig_img_shape[0]), mode='bilinear', align_corners=False)


    # Calculate energy map for "classic" seam carving based on gradients
    tensor_transform = transforms.ToTensor()
    energy_map = calc_energy(tensor_transform(orig_image).unsqueeze(0))
    energy_map = energy_map.squeeze().detach().cpu().numpy()
    #Normalize energy map
    energy_map = (energy_map -energy_map.min())/ (energy_map.max() - energy_map.min() + 1e-8)


    # Estimate the depth of the image
    print("\nEstimating the depth of the image...")
    depth_estimate = depth_estimator.estimate_depth(image_path)
    # # Normalize the depth estimate to range 0-1
    depth_estimate = (depth_estimate - depth_estimate.min()) / (depth_estimate.max() - depth_estimate.min()+1e-8)


    ## Creating a combined costmap from the inpainted feature map and the depth estimate
    # Both the inpainted feature map and the depth estimate are in the range 0-1
    # # Convert the heatmap to a numpy array
    heatmap_inpainted_np = inpainted_heatmap.squeeze().detach().cpu().numpy()

    # Combine energy map and gradcam map with weighing factor
    weight_grad = 0.8
    gradcam_energy_map = weight_grad*heatmap_inpainted_np + (1-weight_grad)* energy_map

    # # Create the combined cost map with weighing factor and scale it to 0-255
    # Maximimum (255) can only be reached if the it is of highest gradCam closest to the camera
    cost_map = ((1-weight_depth)*gradcam_energy_map +(weight_depth*depth_estimate))*255


    # Display the combined cost map next to the depth estimate and the inpainted feature map
    display_combined_cost_map(depth_estimate, inpainted_heatmap, energy_map, cost_map, weight_depth)

    # Remove n_seams from the image based on costmap and create a video if create_video is True
    img_seam_rm, removed_seams = remove_seams_from_image(orig_img_cv2, cost_map, n_seams, create_video=create_video)


    # Display the original image and the image with the seam removed
    display_two_images(orig_img_cv2, img_seam_rm, "Original image", f"Image with {n_seams} seams removed")

    # Vectorize the image with triangles
    print("\nVectorizing the image with triangles...")
    vectorized_img = vectorize_with_triangles(img_seam_rm)
    # show the vectorized image and the seam carved image side by side
    display_two_images(img_seam_rm, vectorized_img, f"Image with {n_seams} seams removed", "Vectorized Image with Triangles")

    # Uncarve the vectorized image without averaging to show gaps
    print("\nUncarving the vectorized image...")
    img_all_seams_highlighted = uncarve(vectorized_img, removed_seams, average=False)
    # Display the vectorized image and the uncarved image side by side
    display_two_images(vectorized_img, img_all_seams_highlighted, "Vectorized Image with Triangles", "Uncarved Image")


    # # uncarve with averaging to fill gaps
    # print("\nUncarving the vectorized image with averaging...")
    # img_all_seams_highlighted_avg = uncarve(vectorized_img, removed_seams, average=True)
    # # Display the vectorized image and the uncarved image side by side
    # display_two_images(vectorized_img, img_all_seams_highlighted_avg, "Vectorized Image with Triangles", "Uncarved Image with Averaging")


    ###################################Vectorization approach 2###########################

    # vectorized_triangles = vectorize(img_seam_rm)
    # display_vectorized_image(img_seam_rm, vectorized_triangles)
    # adjusted_triangles = adjust_triangles_for_uncarve(vectorized_triangles, removed_seams)
    # display_vectorized_image(img_all_seams_highlighted, adjusted_triangles)

    ########################################################



    
