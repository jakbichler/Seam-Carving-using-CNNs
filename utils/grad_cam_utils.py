import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch




def overlay_heatmap(heatmap, image_path):
    """
    Overlay the heatmap on the original image to paint the featuremap.

    Parameters:
    - heatmap: Heatmap to overlay
    - image_path: Path to the original image

    Returns:
    - superimposed_img: Superimposed image
    """

    img = cv2.imread(image_path)

    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite('./map.jpg', superimposed_img)

    return superimposed_img


def modify_features(image, heatmap, image_path, orig_img_shape):
    """
    Modify the featuremap by painting on it.

    Parameters:
    - image: Input image
    - heatmap: Heatmap to modify
    - image_path: Path to the original image
    - orig_img_shape: Shape of the original image

    Returns:
    - heatmap_new: Modified heatmap
    """

    # Convert the image to numpy array
    plt_image = image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
    coords = []
    heatmap_new = heatmap.clone()

    # Global flag for painting
    is_painting = False


    def update_heatmap(event, ix, iy):
        # Convert the coordinates to heatmap coordinates
        ix_heat = ix / orig_img_shape[0] * heatmap_new.shape[0]
        iy_heat = iy / orig_img_shape[1] * heatmap_new.shape[1]

        print(f'Painted at: (x: {ix:.1f}, y: {iy:.1f}, corresponding to heatmap coordinates: (x: {int(ix_heat)}, y: {int(iy_heat)})')

        if event.button == 1:
            # Increase the heatmap value at the painted location
            if heatmap_new[int(iy_heat), int(ix_heat)] < 0.8:
                heatmap_new[int(iy_heat), int(ix_heat)] += 0.2
            else:
                heatmap_new[int(iy_heat), int(ix_heat)] = 1

        if event.button == 3:
            # Decrease the heatmap value at the painted location
            if heatmap_new[int(iy_heat), int(ix_heat)] > 0.2:
                heatmap_new[int(iy_heat), int(ix_heat)] -= 0.2
            else:
                heatmap_new[int(iy_heat), int(ix_heat)] = 0

        # Update the image
        imshow.set_data(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)),  cv2.COLOR_BGR2RGB))
        fig.canvas.draw()

    def on_press(event):
        # Activate painting
        nonlocal is_painting
        is_painting = True
        update_heatmap(event, event.xdata, event.ydata)

    def on_release(event):
        # Deactivate painting
        nonlocal is_painting
        is_painting = False

    def on_motion(event):
        # Update the heatmap while painting
        if is_painting:
            update_heatmap(event, event.xdata, event.ydata)

    fig = plt.figure("Impaint featuremap")
    ax = fig.add_subplot(111)

    # Setup the canvas
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    os.system("clear")
    print ("\nModify the featuremap by painting...\n")

    overlayed_inpainted_image = cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path))

    # Show the image
    imshow = ax.imshow(cv2.cvtColor(overlayed_inpainted_image, cv2.COLOR_BGR2RGB))
    plt.show()

    return heatmap_new