import cv2
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from icecream import ic
from numba import jit


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

    # heatmap_new = F.interpolate(heatmap_new.unsqueeze(0).unsqueeze(0), size=orig_img_shape, mode='bilinear', align_corners=False)


    def onclick(event):
        ix, iy = event.xdata, event.ydata

        # Modify the heatmap
        ic(plt_image.shape)
        # rescale ix and iy to fit heatmap size
        ix_heat = ix / orig_img_shape[0] * heatmap_new.shape[0]
        iy_heat = iy / orig_img_shape[1] * heatmap_new.shape[1]

        print(f'Painted at: (x: {ix:.1f}, y: {iy:.1f}, corresponding to heatmap coordinates: (x: {int(ix_heat)}, y: {int(iy_heat)})')

        # Clip heatmap to max 1
        if heatmap_new[int(iy_heat), int(ix_heat)] < 0.8:
            heatmap_new[int(iy_heat), int(ix_heat)] += 0.2

        else:
            heatmap_new[int(iy_heat), int(ix_heat)] = 1


        # Update the displayed image with the modified heatmap
        imshow.set_data(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)),  cv2.COLOR_BGR2RGB))
        fig.canvas.draw()

    fig = plt.figure("Impaint featuremap")
    ax = fig.add_subplot(111)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    

    overlayed_inpainted_image = cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path))
    overlayed_inpainted_image = cv2.convertScaleAbs(overlayed_inpainted_image)

    imshow = ax.imshow(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)), cv2.COLOR_BGR2RGB))
    plt.show()

    return heatmap_new



# Inspired by https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
@jit
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
def carve_column(img, heatmap):
    r, c, _ = img.shape

    M, backtrack = get_seam(heatmap)

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
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))

    return img, heatmap_new




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

    ic(img.shape)
    r, c, _ = img.shape
    mask = np.zeros((r, c), dtype=bool)

    highlighted_image = img.copy()


    # Find the starting point with the lowest energy in the last row
    j = np.argmin(M[-1])
    for i in range(r - 1, -1, -1):
        mask[i, j] = True
        j = backtrack[i, j]

    # Highlight the seam in red, the image ix width, height, 3 channels. red to 255

        highlighted_image[mask == True, 0] = 0
        highlighted_image[mask == True, 1] = 165
        highlighted_image[mask == True, 2] = 200

    return highlighted_image