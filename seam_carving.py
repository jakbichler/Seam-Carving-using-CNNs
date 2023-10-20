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


    # Resize the heatmap to twice the size for better painitng experience
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(int(1.5 * heatmap.shape[0]), int(1.5 * heatmap.shape[1])), mode='bilinear', align_corners=False)

    # Modify the feature map by painting on it (value range is 0-1)
    inpainted_heatmap = modify_features(img, heatmap.squeeze(), image_path, orig_img_shape)

    # Resize the heatmap to the original image size
    inpainted_heatmap = F.interpolate(inpainted_heatmap.unsqueeze(0).unsqueeze(0), size=(orig_img_shape[1], orig_img_shape[0]), mode='bilinear', align_corners=False)



    ################################################ DEPTH ESTIMATION #############################################################

    # Estimate the depth of the image
    print("\nEstimating the depth of the image...")
    depth_estimate = depth_estimator.estimate_depth(image_path)

    ################################################################################################################


    ## Creating a combined costmap from the inpainted feature map and the depth estimate
    # Both the inpainted feature map and the depth estimate are in the range 0-1

    # # Convert the heatmap to a numpy array
    heatmap_inpainted_np = inpainted_heatmap.squeeze().detach().cpu().numpy()

    # # Normalize the depth estimate to range 0-1
    depth_estimate = (depth_estimate - depth_estimate.min()) / (depth_estimate.max() - depth_estimate.min()+1e-8)


    # # Create the combined cost map with weighing factor and scale it to 0-255
    cost_map = ((1-weight_depth)*heatmap_inpainted_np + weight_depth*depth_estimate)*255




    # # Display the combined cost map next to the depth estimate and the inpainted heatmap
    # Create main axes: one for left half and one for right half
    # Create main axes: one for the right half and one for the left half
    fig = plt.figure(figsize=(20, 10))
    ax_right = plt.subplot2grid((2, 2), (0, 1), rowspan=2, fig=fig)
    ax_left_top = plt.subplot2grid((2, 2), (0, 0), fig=fig)
    ax_left_bottom = plt.subplot2grid((2, 2), (1, 0), fig=fig)

    # Depth Estimate with colorbar in top-left
    im_depth = ax_left_top.imshow(depth_estimate)
    ax_left_top.set_title("Depth Estimate")
    cbar_depth = fig.colorbar(im_depth, ax=ax_left_top, orientation='vertical')
    cbar_depth.set_label('Inverse Depth Value', rotation=270, labelpad=15)

    # Inpainted Heatmap with colorbar in bottom-left
    im_heatmap = ax_left_bottom.imshow(inpainted_heatmap.squeeze())
    ax_left_bottom.set_title("Inpainted GradCam Feature Map")
    cbar_heatmap = fig.colorbar(im_heatmap, ax=ax_left_bottom, orientation='vertical')
    cbar_heatmap.set_label('Heatmap Value', rotation=270, labelpad=15)

    # Combined Cost Map with colorbar on the right, spanning two rows
    im_cost = ax_right.imshow(cost_map)
    ax_right.set_title(f"Combined Cost Map with depth_weight {weight_depth}")
    cbar_cost = fig.colorbar(im_cost, ax=ax_right, orientation='vertical', shrink= 0.6)
    cbar_cost.set_label('Cost Value', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()



    # Remove the seam from the image
    img_seam_rm = orig_img_cv2.copy()
    heatmap_seam_removed = cost_map.copy()

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



    
