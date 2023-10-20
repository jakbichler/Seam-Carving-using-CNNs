import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from icecream import ic
import cv2
import argparse
from PIL import Image
from models.vgg19.vgg import CustomVGG
from utils import *
import os


    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--class_id', type=int, help='Class to perform Grad-CAM with')
    parser.add_argument('--n_seams', type=int, help='Number of seams to remove')
    parser.add_argument('--create_video', action='store_true', help='Whether to create a video or not')
    args = parser.parse_args()

    image_path = args.image_path
    class_id = args.class_id
    n_seams = args.n_seams
    create_video = args.create_video



    # Initialize the CustomVGG model
    model = CustomVGG()
    model.eval()

    # Load an image
    orig_image = Image.open(image_path)
    orig_img_cv2 = cv2.imread(image_path)


    # save the shape of the image
    orig_img_shape = orig_image.size

    # Transform and preprocess the image
    img = torch.FloatTensor(model.transform(orig_image).unsqueeze(0))

    # Generate the Grad-CAM heatmap
    heatmap = model.grad_cam_heatmap(class_id, img)


    # Resize the heatmap to twice the size for better painitng experience
    # heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(2 * heatmap.shape[0], 2 * heatmap.shape[1]), mode='bilinear', align_corners=False)

    # Modify the feature map by painting on it (value range is 0-1)
    inpainted_heatmap = modify_features(img, heatmap.squeeze(), image_path, orig_img_shape)

    # Resize the heatmap to the original image size
    inpainted_heatmap = F.interpolate(inpainted_heatmap.unsqueeze(0).unsqueeze(0), size=(orig_img_shape[1], orig_img_shape[0]), mode='bilinear', align_corners=False)

    
    # Plot the heatmap
    fig, ax = plt.subplots()  # Create a new figure and axis object
    cax = ax.matshow(inpainted_heatmap.squeeze())
    plt.title("Guide for seam_carving")

    # Add a colorbar to the figure to act as a legend
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.6, 0.9], orientation='vertical')
    cbar.set_label('Value', rotation=270, labelpad=30)

    plt.show()


    ## Implementation of seam_carving
    # # Convert the heatmap to a numpy array
    heatmap_np = inpainted_heatmap.squeeze().detach().cpu().numpy() * 255

    # Calculate the first seam
    print("\nCalculating the first seam...\n")
    M, backtrack = get_seam(heatmap_np)
    
    heatmap_np = np.uint8(heatmap_np)

    # display the original image with the highlighted seam
    image_highlighted_seam = highlight_seam_image(orig_img_cv2, M, backtrack)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(image_highlighted_seam),  cv2.COLOR_BGR2RGB))
    plt.title("Original image with the first seam to be removed highlighted")
    plt.show()



    # Remove the seam from the image
    img_seam_rm = orig_img_cv2.copy()
    heatmap_seam_removed = heatmap_np.copy()

    removed_seams = []

    # Create a tqdm progress bar instance
    pbar = tqdm(total=n_seams, desc=f"Removing {n_seams} Seams", dynamic_ncols=True, leave=True)


    if create_video:
        # Create a video
        out = cv2.VideoWriter("video_carving.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=3, frameSize=(img_seam_rm.shape[1], img_seam_rm.shape[0]))


        # Visualizing the video of seam_carving
        for i in range(n_seams):

            # Calculate the seam for the current heatmap
            M, backtrack = get_seam(heatmap_seam_removed)

            # Highlight the current seam on the image
            highlighted_img = highlight_seam_image(img_seam_rm, M, backtrack)

            # Write the highlighted image to the video
            img_padded_highlighted = np.zeros((orig_img_cv2.shape[0], orig_img_cv2.shape[1], 3), dtype=int)
            img_padded_highlighted[:, :highlighted_img.shape[1], :] = highlighted_img
            out.write(cv2.convertScaleAbs(img_padded_highlighted))

            mask_seam, img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed, M, backtrack)

            removed_seams.append(mask_seam)

            # Update tqdm bar
            pbar.update(1)


        # Close the tqdm progress bar
        pbar.close()
        out.release()



        # No video, just carving
    else:

        for i in range(n_seams):

            # Calculate the seam for the current heatmap
            M, backtrack = get_seam(heatmap_seam_removed)

            mask_seam, img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed, M, backtrack)
  
            removed_seams.append(mask_seam)
            pbar.update(1)


        pbar.close()


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



    
