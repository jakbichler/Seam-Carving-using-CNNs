import cv2
import numpy as np
import matplotlib.pyplot as plt 
from icecream import ic


def overlay_heatmap(heatmap, image_path):
    img = cv2.imread(image_path)

    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite('./map.jpg', superimposed_img)

    return superimposed_img



# def modify_features(image, heatmap, image_path):
    
#     plt_image = image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
#     coords = []
#     heatmap_new = heatmap.clone()

#     def onclick(event):
#         ix, iy = event.xdata, event.ydata
#         print(f'Painted at: (x: {ix}, y: {iy})')

#         # Modify the heatmap
#         heatmap_new[int(iy), int(ix)] += 0.1


#         # Update the displayed image with the modified heatmap
#         imshow.set_data(heatmap_new.squeeze().detach().cpu().numpy())
#         fig.canvas.draw()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
#     imshow = ax.imshow(heatmap_new.squeeze().detach().cpu().numpy())
#     plt.show()

#     return heatmap_new



def modify_features(image, heatmap, image_path, orig_img_shape):
    
    plt_image = image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
    coords = []
    heatmap_new = heatmap.clone()

    def onclick(event):
        ix, iy = event.xdata, event.ydata


        # Modify the heatmap
        ic(plt_image.shape)
        # rescale ix and iy to fit heatmap size
        ix = ix / orig_img_shape[0] * heatmap_new.shape[1]
        iy = iy / orig_img_shape[1] * heatmap_new.shape[0]

        print(f'Painted at: (x: {ix}, y: {iy})')

        heatmap_new[int(iy), int(ix)] += 0.1


        # Update the displayed image with the modified heatmap
        imshow.set_data(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)),  cv2.COLOR_BGR2RGB))
        fig.canvas.draw()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    

    overlayed_inpainted_image = cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path))
    overlayed_inpainted_image = cv2.convertScaleAbs(overlayed_inpainted_image)

    imshow = ax.imshow(cv2.cvtColor(cv2.convertScaleAbs(overlay_heatmap(heatmap_new, image_path)), cv2.COLOR_BGR2RGB))
    plt.show()

    return heatmap_new