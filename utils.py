import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from icecream import ic
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F





def display_two_images(img1, img2, title1, title2, figsize=(20, 10)):
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
    cbar_cost = fig.colorbar(im_cost, ax=ax_right, orientation='vertical', shrink=0.8)
    cbar_cost.set_label('Cost Value', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()




def overlay_heatmap(heatmap, image_path):
    img = cv2.imread(image_path)

    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite('./map.jpg', superimposed_img)

    return superimposed_img


def modify_features(image, heatmap, image_path, orig_img_shape):
    plt_image = image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
    coords = []
    heatmap_new = heatmap.clone()

    # Global flag for painting
    is_painting = False



    def update_heatmap(event, ix, iy):
        ix_heat = ix / orig_img_shape[0] * heatmap_new.shape[0]
        iy_heat = iy / orig_img_shape[1] * heatmap_new.shape[1]

        print(f'Painted at: (x: {ix:.1f}, y: {iy:.1f}, corresponding to heatmap coordinates: (x: {int(ix_heat)}, y: {int(iy_heat)})')

        if event.button == 1:
            if heatmap_new[int(iy_heat), int(ix_heat)] < 0.8:
                heatmap_new[int(iy_heat), int(ix_heat)] += 0.2
            else:
                heatmap_new[int(iy_heat), int(ix_heat)] = 1

        if event.button == 3:
            if heatmap_new[int(iy_heat), int(ix_heat)] > 0.2:
                heatmap_new[int(iy_heat), int(ix_heat)] -= 0.2
            else:
                heatmap_new[int(iy_heat), int(ix_heat)] = 0

        imshow.set_data(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)),  cv2.COLOR_BGR2RGB))
        fig.canvas.draw()

    def on_press(event):
        nonlocal is_painting
        is_painting = True
        update_heatmap(event, event.xdata, event.ydata)

    def on_release(event):
        nonlocal is_painting
        is_painting = False

    def on_motion(event):
        if is_painting:
            update_heatmap(event, event.xdata, event.ydata)

    fig = plt.figure("Impaint featuremap")
    ax = fig.add_subplot(111)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    os.system("clear")
    print ("\nModify the featuremap by painting...\n")

    overlayed_inpainted_image = cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path))

    imshow = ax.imshow(cv2.cvtColor(overlayed_inpainted_image, cv2.COLOR_BGR2RGB))
    plt.show()

    return heatmap_new




# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
def get_seam(heatmap):
    r, c = heatmap.shape


    M = heatmap.copy().astype(np.uint32)
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack




# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
def carve_column(img, heatmap, M, backtrack):
    r, c, _ = img.shape

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Remove seam from heatmap
    heatmap_new = heatmap[mask].reshape((r, c - 1))


    # Remove seam from image
    mask_stacked = np.stack([mask] * 3, axis=2)
    img = img[mask_stacked].reshape((r, c - 1, 3))

    return mask, img, heatmap_new



# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
# But completelty changes to pytorch functionality
def calc_energy(img_tensor):
    # Define the filters
    filter_du = torch.tensor([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 3, 3]

    filter_dv = torch.tensor([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 3, 3]

    # Expand filters to 3 input channels (R, G, B)
    filter_du = filter_du.repeat(1, 3, 1, 1)
    filter_dv = filter_dv.repeat(1, 3, 1, 1)

    # Convolve the image with the filters
    convolved_du = F.conv2d(img_tensor, filter_du, padding=1)
    convolved_dv = F.conv2d(img_tensor, filter_dv, padding=1)
    
    convolved = torch.abs(convolved_du) + torch.abs(convolved_dv)
    
    # Sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(dim=1, keepdim=True)

    return energy_map





def remove_seams_from_image(orig_img_cv2, cost_map, n_seams, create_video=False):
    """
    Remove seams from the given image.

    Parameters:
    - orig_img_cv2: Original image
    - cost_map: Cost map used for seam carving
    - n_seams: Number of seams to remove
    - get_seam: Function to calculate seam from the heatmap
    - carve_column: Function to carve a column from the image and heatmap
    - create_video (optional): Flag to create a seam carving video
    """
    
    img_seam_rm = orig_img_cv2.copy()
    heatmap_seam_removed = cost_map.copy()

    removed_seams = []

    pbar = tqdm(total=n_seams, desc=f"Removing {n_seams} Seams", dynamic_ncols=True, leave=True)

    if create_video:
        out = cv2.VideoWriter("video_carving.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=3, frameSize=(img_seam_rm.shape[1], img_seam_rm.shape[0]))

        for i in range(n_seams):
            M, backtrack = get_seam(heatmap_seam_removed)
            highlighted_img = highlight_seam_image(img_seam_rm, M, backtrack)

            img_padded_highlighted = np.zeros((orig_img_cv2.shape[0], orig_img_cv2.shape[1], 3), dtype=int)
            img_padded_highlighted[:, :highlighted_img.shape[1], :] = highlighted_img
            out.write(cv2.convertScaleAbs(img_padded_highlighted))

            mask_seam, img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed, M, backtrack)
            removed_seams.append(mask_seam)
            pbar.update(1)

        pbar.close()
        out.release()

    else:
        for i in range(n_seams):
            M, backtrack = get_seam(heatmap_seam_removed)
            mask_seam, img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed, M, backtrack)
            removed_seams.append(mask_seam)
            pbar.update(1)

        pbar.close()

    return img_seam_rm, removed_seams



def highlight_seam_heatmap(img, M, backtrack):
    r, c = img.shape
    mask = np.zeros((r, c), dtype=bool)
 

    highlighted_heatmap = img.copy()


    # Find the starting point with the lowest energy in the last row
    j = np.argmin(M[-1])
    for i in range(r - 1, -1, -1):
        mask[i, j] = True
        j = backtrack[i, j]

    # Highlight the seam 
    highlighted_heatmap[mask == True] = 255  

    return highlighted_heatmap



def highlight_seam_image(img, M, backtrack):

    r, c, _ = img.shape
    mask = np.zeros((r, c), dtype=bool)

    highlighted_image = img.copy()


    # Find the starting point with the lowest energy in the last row
    j = np.argmin(M[-1])
    for i in range(r - 1, -1, -1):
        mask[i, j] = True
        j = backtrack[i, j]

    # Highlight the seam in red, the image ix width, height, 3 channels. red to 255

        highlighted_image[mask == True, 0] = 200
        highlighted_image[mask == True, 1] = 0
        highlighted_image[mask == True, 2] = 0

    return highlighted_image




def show_missing_seams(carved_image, masks):
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



def generate_vertices(image):
    """ 
    Generate vertices for the image.
    """
    height, width = image.shape[:2]
    y, x = np.mgrid[:height, :width]
    vertices = np.column_stack((x.ravel(), y.ravel()))

    return vertices



def generate_triangles(image):
    """
    Generate triangles for the image.
    """
    height, width = image.shape[:2]
    triangles = []

    for y in range(height-1):
        for x in range(width-1):

            top_left = y * (width ) + x
            top_right = top_left + 1

            bottom_left = (y + 1) * (width) + x
            bottom_right = bottom_left + 1

            triangles.append([top_left, top_right, bottom_right])
            triangles.append([top_left, bottom_right, bottom_left])

    return triangles



def insert_removed_vertices(vertices, removed_seams):
    """
    Insert the removed seams back into the image, 
    by shifting vertices of the vecotrized image.
    """
    updated_vertices = []

    # Loop through the removed seams in reversed order
    for seam in tqdm(reversed(removed_seams), desc="Reintroducing removed seams", unit="seam"):


        seam_positions = np.where(seam==False)
        seam_rows, seam_cols = seam_positions

        # Temporary storage to hold updated vertices for this iteration
        temp_vertices = []

        for vertex in vertices:
            col, row = vertex
            adjusted_col = col
            
            # Use zip to iterate over seam rows and columns simultaneously
            for s_row, s_col in zip(seam_rows, seam_cols):
                # Only look at vertices on the same row as the current seam row
                if row == s_row:
                    # If the vertex's column is to the right of the reintroduced seam, shift it to the right
                    if col >= s_col:
                        adjusted_col += 1

            # Append the possibly adjusted vertex to the temporary list
            temp_vertices.append([adjusted_col, row])
            
        # Set vertices to the updated vertices for this iteration
        vertices = temp_vertices
    
    # After processing all seams, set the final list of vertices
    updated_vertices = vertices
    
    return updated_vertices



def visualize_grid(image, vertices, stretched_vertices, triangles):
    """
    Visualize the grid for the original vertices and the stretched one side-by-side.
    """


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


    for tri in tqdm(triangles, desc = "Plotting triangles", unit = "triangle"):
        tri_vertices = np.array([vertices[i] for i in tri])
        tri_vertices_str = np.array([stretched_vertices[i] for i in tri])
        polygon = patches.Polygon(tri_vertices, fill=None, edgecolor='r', closed=True)
        polygon_stretched = patches.Polygon(tri_vertices_str, fill=None, edgecolor='r', closed=True)
        ax1.add_patch(polygon)
        ax2.add_patch(polygon_stretched)
    
        # plot vertices as small dots
        ax1.plot(tri_vertices[:, 0], tri_vertices[:, 1], 'o', color='b', markersize=1)
        ax2.plot(tri_vertices_str[:, 0], tri_vertices_str[:, 1], 'o', color='b', markersize=1)

    height, width = image.shape[:2]
    ax1.set_xlim([-2, width + 1])
    ax1.set_ylim([height + 2, -1])
    ax1.imshow(image, aspect='auto')
    ax2.set_xlim([-2, width + 2])
    ax2.set_ylim([height + 2, -1])
    ax2.imshow(image, aspect='auto')
    plt.show()


