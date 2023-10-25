import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

def display_two_images(img1, img2, title1, title2, figsize=(20, 10), save_path = None):
    """
    Display two images side by side using matplotlib.

    :param img1: First image (numpy array)
    :param img2: Second image (numpy array)
    :param title1: Title for the first image
    :param title2: Title for the second image
    :param figsize: Tuple specifying the figure size (optional)
    """
    plt.figure(figsize=figsize)
    
    # Display first image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(img1), cv2.COLOR_BGR2RGB))
    plt.title(title1)
    
    # Display second image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(img2), cv2.COLOR_BGR2RGB))
    plt.title(title2)
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



def display_combined_cost_map(depth_estimate, inpainted_heatmap, energy_map, cost_map, weight_depth):
    """
    Display the combined cost map next to the depth estimate and the inpainted heatmap.

    Parameters:
    - depth_estimate: 2D numpy array or similar representing depth estimate
    - inpainted_heatmap: 2D numpy array or similar representing inpainted heatmap
    - cost_map: 2D numpy array or similar representing combined cost map
    - weight_depth: weight parameter to be used for display title
    - third_image: 2D numpy array or similar representing the third image to be displayed
    """
    fig = plt.figure(figsize=(20, 10))
    ax_right = plt.subplot2grid((3, 2), (0, 1), rowspan=3, fig=fig)
    ax_left_top = plt.subplot2grid((3, 2), (0, 0), fig=fig)
    ax_left_middle = plt.subplot2grid((3, 2), (1, 0), fig=fig)
    ax_left_bottom = plt.subplot2grid((3, 2), (2, 0), fig=fig)

    # Depth Estimate with colorbar in top-left
    im_depth = ax_left_bottom.imshow(depth_estimate)
    ax_left_bottom.set_title("Depth Estimate (MiDaS CNN)")
    cbar_depth = fig.colorbar(im_depth, ax=ax_left_bottom, orientation='vertical')
    cbar_depth.set_label('Inverse Depth Value', rotation=270, labelpad=15)


    # Third Image with colorbar in middle-left
    im_third = ax_left_middle.imshow(energy_map)
    ax_left_middle.set_title("Energy Map (Gradients in x,y)")
    cbar_third = fig.colorbar(im_third, ax=ax_left_middle, orientation='vertical')
    cbar_third.set_label('Energy Map Value', rotation=270, labelpad=15)


    # Inpainted Heatmap with colorbar in bottom-left
    im_heatmap = ax_left_top.imshow(inpainted_heatmap.squeeze())
    ax_left_top.set_title("Inpainted GradCam Feature Map (VGG CNN)")
    cbar_heatmap = fig.colorbar(im_heatmap, ax=ax_left_top, orientation='vertical')
    cbar_heatmap.set_label('Heatmap Value', rotation=270, labelpad=15)


    # Combined Cost Map with colorbar on the right, spanning two rows
    im_cost = ax_right.imshow(cost_map)
    ax_right.set_title(f"Combined Cost Map used for Seam Carving")
    cbar_cost = fig.colorbar(im_cost, ax=ax_right, orientation='vertical', shrink=0.58)
    cbar_cost.set_label('Cost Value', rotation=270, labelpad=15)


    # Save the plt figure
    plt.savefig("outputs/combined_cost_map.png", dpi=500, bbox_inches='tight')
    plt.tight_layout()
    plt.show()



def show_missing_cols(carved_image, masks):
    img = np.array(carved_image)

    pbar = tqdm(total=len(masks), desc="Highlighting removed seams", dynamic_ncols=True, leave=True)
    
    for mask in reversed(masks):

        new_img = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]), dtype=img.dtype)
        
        for row in range(mask.shape[0]):

            # Find the index where the mask in that row is zero
            # This is the index of the pixel that was removed
            # We will replace this pixel with the average of the two neighbouring pixels
            pixel_index = np.where(mask[row] == 0)[0][0]

            # Where mask is 0, leave the new image to zero, fill the new image with old image 
            # left and right of the pixel that was removed
            new_img[row, :pixel_index] = img[row, :pixel_index]
            new_img[row, pixel_index+1:] = img[row, pixel_index:]
            new_img[row, pixel_index] = 0  
    
        pbar.update(1)
        img = new_img

    pbar.close()

    return img


def show_missing_rows(carved_image, removed_rows):
    img = np.array(carved_image)

    pbar = tqdm(total=len(removed_rows), desc="Highlighting removed seams", dynamic_ncols=True, leave=True)
    
    for mask in reversed(removed_rows):

        new_img = np.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]), dtype=img.dtype)
        
        for col in range(mask.shape[1]):

            # Find the index where the mask in that row is zero
            # This is the index of the pixel that was removed
            # We will replace this pixel with the average of the two neighbouring pixels
            pixel_index = np.where(mask[:,col] == 0)[0][0]


            # Where mask is 0, leave the new image to zero, fill the new image with old image 
            # left and right of the pixel that was removed
            new_img[:pixel_index, col] = img[:pixel_index, col]
            new_img[pixel_index +1:, col] = img[pixel_index:, col]
            new_img[pixel_index, col] = 0  
    
        pbar.update(1)
        img = new_img

    pbar.close()

    return img
