import cv2
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm



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

    overlayed_inpainted_image = cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path))

    imshow = ax.imshow(cv2.cvtColor(overlayed_inpainted_image, cv2.COLOR_BGR2RGB))
    plt.show()

    return heatmap_new




# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
def get_seam(heatmap):
    r, c = heatmap.shape

    M = heatmap.copy()
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



def vectorize_with_triangles(image):
    # Create a blank canvas to draw the triangles
    canvas = np.zeros_like(image)
    
    # Define the triangle-drawing function using OpenCV
    def draw_triangle(img, pts, color):
        triangle_cnt = np.array(pts).reshape((-1,1,2))
        cv2.drawContours(img, [triangle_cnt], 0, color, -1)

    # Iterate through the image pixels in a 2x2 block manner
    for i in tqdm(range(0, image.shape[1]-1, 2)):
        for j in range(0, image.shape[0]-1, 2):
            # Centers of the 4 pixels in the block
            centers = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
            
            # Define the two triangles using the centers
            triangle1 = [centers[0], centers[1], centers[2]]
            triangle2 = [centers[1], centers[2], centers[3]]

            # Get average color for each triangle from the image
            color1 = np.mean([image[y,x] for x,y in triangle1], axis=0)
            color2 = np.mean([image[y,x] for x,y in triangle2], axis=0)

            # Draw the triangles onto the canvas
            draw_triangle(canvas, triangle1, color1)
            draw_triangle(canvas, triangle2, color2)
            
    return canvas


# 1. Using Averages
def uncarve_using_averages(carved_image, masks):
    img = np.array(carved_image)

    for mask in tqdm(reversed(masks)):

        new_img = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]), dtype=img.dtype)
        
        for row in range(mask.shape[0]):

            # Find the index where the mask in that row is zero
            # This is the index of the pixel that was removed
            # We will replace this pixel with the average of the two neighbouring pixels
            pixel_index = np.where(mask[row] == 0)[0][0]

            # Where mask is 0, leave the new image to zero, fill the new image with old image 
            # left and right of the pixel that was removed
            new_img[row, :pixel_index] = img[row, :pixel_index]
            new_img[row, pixel_index] = 0  
            new_img[row, pixel_index+1:] = img[row, pixel_index:]

        

        img = new_img
    return img