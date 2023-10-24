import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


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
    r, c = img.shape[:2]

    # Create a (r, c) matrix filled with the value True
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since you're removing one column (seam) from the image, 
    # you need to adjust the mask accordingly for multi-channel images
    mask_rgb = np.stack([mask] * 3, axis=2)

    # Remove seam from image and heatmap
    img_new = img[mask_rgb].reshape((r, c - 1, 3))
    heatmap_new = heatmap[mask].reshape((r, c - 1))

    return mask, img_new, heatmap_new



# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
# But completelty changed to pytorch functionality
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



def remove_seams_from_image(orig_img_cv2, cost_map, n_cols, n_rows, create_video=False):
    """
    Remove seams from the given image both horizontally and vertically.

    Parameters:
    - orig_img_cv2: Original image
    - cost_map: Cost map used for seam carving
    - n_cols: Number of vertical seams to remove
    - n_rows: Number of horizontal seams to remove
    - create_video (optional): Flag to create a seam carving video
    """
    
    img_seam_rm = orig_img_cv2.copy()
    heatmap_seam_removed = cost_map.copy()

    removed_seams = []

    if create_video:
        out = cv2.VideoWriter("outputs/video_carving.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=3, frameSize=(orig_img_cv2.shape[1], orig_img_cv2.shape[0]))
    else:
        out = None

    for idx, (num_seams, remove_horizontal) in enumerate([(n_cols, False), (n_rows, True)]):
        if remove_horizontal:
            # Rotate the image and heatmap by 90 degrees to remove horizontal seams
            img_seam_rm = cv2.rotate(img_seam_rm, cv2.ROTATE_90_CLOCKWISE)
            heatmap_seam_removed = cv2.rotate(heatmap_seam_removed, cv2.ROTATE_90_CLOCKWISE)

        pbar = tqdm(total=num_seams, desc=f"Removing {num_seams} {'horizontal' if remove_horizontal else 'vertical'} Seams", dynamic_ncols=True, leave=True)
        
        for i in range(num_seams):
            M, backtrack = get_seam(heatmap_seam_removed)
            highlighted_img = highlight_seam_image(img_seam_rm, M, backtrack)
            
            if out:
                if remove_horizontal:

                    img_padded_highlighted = np.zeros((orig_img_cv2.shape[1], orig_img_cv2.shape[0], 3), dtype=int)
                    img_padded_highlighted[:highlighted_img.shape[0], :highlighted_img.shape[1], :] = highlighted_img

                    out.write(cv2.convertScaleAbs(cv2.rotate(img_padded_highlighted, cv2.ROTATE_90_COUNTERCLOCKWISE)))

                else:
                    img_padded_highlighted = np.zeros((orig_img_cv2.shape[0], orig_img_cv2.shape[1], 3), dtype=int)
                    img_padded_highlighted[:, :highlighted_img.shape[1], :] = highlighted_img
                    out.write(cv2.convertScaleAbs(img_padded_highlighted))

            
            
            mask_seam, img_seam_rm, heatmap_seam_removed = carve_column(img_seam_rm, heatmap_seam_removed, M, backtrack)
            
            if remove_horizontal:
                removed_seams.append(np.flipud(mask_seam.transpose()))
                
            else:
                removed_seams.append((mask_seam))
            pbar.update(1)

        pbar.close()

        if remove_horizontal:
            # Rotate the carved image back by -90 degrees to restore its original orientation
            img_seam_rm = cv2.rotate(img_seam_rm, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if out:
        out.release()

    return img_seam_rm, removed_seams, heatmap_seam_removed




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
