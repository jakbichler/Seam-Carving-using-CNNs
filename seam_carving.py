import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
from utils import overlay_heatmap, modify_features, get_seam, highlight_seam_heatmap, highlight_seam_image, carve_column

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
    orig_image = Image.open(image_path)
    orig_img_cv2 = cv2.imread(image_path)

    ic(orig_image.size)

    # save the shape of the image
    orig_img_shape = orig_image.size

    # Transform and preprocess the image
    img = torch.FloatTensor(model.transform(orig_image).unsqueeze(0))

    # Generate the Grad-CAM heatmap
    heatmap = model.grad_cam_heatmap(class_id, img)

    # Print number of heatmap entries that are zero
    ic((heatmap == 0).sum())

    # Resize the heatmap to twice the size for better painitng experience
    # heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(2 * heatmap.shape[0], 2 * heatmap.shape[1]), mode='bilinear', align_corners=False)

    # Modify the feature map by painting on it (value range is 0-1)
    inpainted_heatmap = modify_features(img, heatmap.squeeze(), image_path, orig_img_shape)

    # Resize the heatmap to the original image size
    inpainted_heatmap = F.interpolate(inpainted_heatmap.unsqueeze(0).unsqueeze(0), size=(orig_img_shape[1], orig_img_shape[0]), mode='bilinear', align_corners=False)

    
    plt.matshow(inpainted_heatmap.squeeze())
    plt.title("inpainted_heatmap")
    plt.show()




    ## Implementation of seam_carving
    # # Convert the heatmap to a numpy array
    heatmap_np = inpainted_heatmap.squeeze().detach().cpu().numpy()

    # Calculate the first seam
    M, backtrack = get_seam(heatmap_np)
    heatmap_np = np.uint8(255 * heatmap_np)


    # display the heatmap with the highlighted seam
    heatmap_highlighted_seam = highlight_seam_heatmap(heatmap_np, M, backtrack)
    plt.matshow(heatmap_highlighted_seam)
    plt.title("highlighted_heatmap")
    plt.show()


    # display the original image with the highlighted seam
    image_highlighted_seam = highlight_seam_image(orig_img_cv2, M, backtrack)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(image_highlighted_seam),  cv2.COLOR_BGR2RGB))
    plt.title("highlighted_image")
    plt.show()



    # Remove the seam from the image
    ic (orig_img_cv2.shape)
    img_seam_rm = orig_img_cv2.copy()
    ic(type (orig_img_cv2))

    heatmap_seam_removed = heatmap_np.copy()

    # Create a video
    out = cv2.VideoWriter("video_carving.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=5, frameSize=(img_seam_rm.shape[1], img_seam_rm.shape[0]))


    # Visualizing the video of seam_carving
    for i in tqdm(range(100)):
        img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed)

        img_padded = np.zeros((orig_img_cv2.shape[0], orig_img_cv2.shape[1], 3), dtype=int)
        img_padded[:,:img_seam_rm.shape[1], :] = img_seam_rm
        ic(img_padded.shape)

        # Inside the loop where you write images to the video
        out.write(cv2.convertScaleAbs(img_padded))

        
    out.release()


    # show the seam carved image and the original side by side
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(orig_img_cv2),  cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(img_seam_rm),  cv2.COLOR_BGR2RGB))
    plt.title("Image with seam removed")

    plt.show()

    ic(img_seam_rm.shape)