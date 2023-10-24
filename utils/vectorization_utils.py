import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from tqdm import tqdm



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
    Generate triangles for the image along with their interpolated colors.
    """
    height, width = image.shape[:2]
    triangles = []
    triangle_colors = {}

    for y in range(height-1):
        for x in range(width-1):

            top_left = y * width + x
            top_right = top_left + 1
            bottom_left = (y + 1) * width + x
            bottom_right = bottom_left + 1

            triangle1 = [top_left, top_right, bottom_right]
            triangle2 = [top_left, bottom_right, bottom_left]

            # Get vertices of the triangles in (x, y) format
            vertices1 = [(x, y), (x + 1, y), (x + 1, y + 1)]
            vertices2 = [(x, y), (x + 1, y + 1), (x, y + 1)]


            # Store triangles and their colors
            triangles.extend([triangle1, triangle2])



    return triangles




def get_triangle_color(triangle_vertices, image):
    """
    Compute the average color of the triangle in the image.
    
    Parameters:
    - triangle_vertices: List of the triangle's vertices. Each vertex is a tuple (x, y).
    - image: The image from which to sample the color.
    
    Returns:
    - The average color of the triangle in the image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a black mask with the same dimensions as the image.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert triangle vertices to integer and reshape for contour drawing
    triangle_contour = np.array(triangle_vertices).astype(int).reshape((-1, 1, 2))
    
    # Fill the triangle in the mask with white.
    cv2.drawContours(mask, [triangle_contour], 0, (255), -1)
    
    # Compute the average color inside the triangle.
    average_color = cv2.mean(image, mask=mask)[:3]

    # Normalize the average color to be in the range [0, 1]
    normalized_color = tuple([x / 255 for x in average_color])

    return normalized_color



def insert_removed_vertices(vertices, removed_rows, removed_cols):
    """Insert the removed seams back into the image by shifting vertices of the vectorized image."""

    for removed_seams, desc, axis in zip([removed_rows, removed_cols],
                                         ["Reinserting removed seams vertically", "Reinserting removed seams horizontally"],
                                         [0, 1]):

        pbar = tqdm(total=len(removed_seams), desc=desc, unit="seam")

        for seam in reversed(removed_seams):
            seam_positions = np.where(seam == False)
            seam_rows, seam_cols = seam_positions
            temp_vertices = []

            for vertex in vertices:
                col, row = vertex
                
                # Adjusting row
                if axis == 0:
                    for s_row, s_col in zip(seam_rows, seam_cols):
                        if col == s_col and row >= s_row:
                            row += 1

                # Adjusting col
                if axis == 1:
                    for s_row, s_col in zip(seam_rows, seam_cols):
                        if row == s_row and col >= s_col:
                            col += 1

                temp_vertices.append([col, row])

            vertices = temp_vertices
            pbar.update(1)

        pbar.close()

    return vertices



def visualize_grid(image, vertices, stretched_vertices, triangles, n_seams):
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
    ax1.set_title("Original")
    ax1.set_xlim([-2, width + 1])
    ax1.set_ylim([height + 2, -1])
    ax1.imshow(image, aspect='auto')
    ax2.set_title("Stretched")
    ax2.set_xlim([-2, width + 2 + n_seams])
    ax2.set_ylim([height + 2, -1])
    ax2.imshow(image, aspect='auto')
    plt.show()




def visualize_stretched_graphics(image, vertices, stretched_vertices, triangles, grid=False):
    """
    Visualize the grid for the stretched vector graphics with triangles filled with their respective color.
    """
    fig, ax = plt.subplots(figsize=(6, 6))


    for tri in tqdm(triangles, desc="Plotting colored and stretched triangles", unit="triangle"):
        tri_vertices_original = np.array([vertices[i] for i in tri])
        tri_vertices_stretched = np.array([stretched_vertices[i] for i in tri])
         
        # Get the average color of the triangle in the original image
        color = get_triangle_color(tri_vertices_original, image)

        if grid:

            # Create a colored triangle for the stretched version
            polygon_stretched = patches.Polygon(tri_vertices_stretched, facecolor=color, edgecolor='r', closed=True)
            ax.add_patch(polygon_stretched)
            #Optionally, if you want to see vertices as small dots
            ax.plot(tri_vertices_stretched[:, 0], tri_vertices_stretched[:, 1], 'o', color='b', markersize=1)
        
        else:
            # Create a colored triangle for the stretched version
            polygon_stretched = patches.Polygon(tri_vertices_stretched, facecolor=color, edgecolor='none', closed=True)
            ax.add_patch(polygon_stretched)



    height, width = image.shape[:2]
    ax.set_title("Stretched Vector Graphics")
    ax.set_xlim([-2, width + 2])
    ax.set_ylim([height + 2, -1])
    plt.axis('off')  # To hide axis
    plt.show()





def barycentric(p, a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    p = np.array(p)

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def rasterize(src_img, dst_img, src_tri, dst_tri):
    h, w, _ = dst_img.shape

    # Compute bounding box 
    bbox_top_left = (int(min(pt[0] for pt in dst_tri)), int(min(pt[1] for pt in dst_tri)))
    bbox_bottom_right = (int(max(pt[0] for pt in dst_tri)), int(max(pt[1] for pt in dst_tri)))

    # Iterate over bounding box
    for y in range(bbox_top_left[1], bbox_bottom_right[1] + 1):
        for x in range(bbox_top_left[0], bbox_bottom_right[0] + 1):
            u, v, w = barycentric(np.array([x, y]), dst_tri[0], dst_tri[1], dst_tri[2])
            if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:  # Ensure the point lies within the triangle
                src_x = int(src_tri[0][0] * u + src_tri[1][0] * v + src_tri[2][0] * w)
                src_y = int(src_tri[0][1] * u + src_tri[1][1] * v + src_tri[2][1] * w)
                dst_img[y, x] = src_img[src_y, src_x]



def interpolate_rasterize(src_img, vertices, stretched_vertices, triangles):
    dst_img = np.zeros_like(src_img)



    for tri in tqdm(triangles, desc="Interpolating and rasterizing triangles", unit="triangle"):
        src_tri = [vertices[i] for i in tri]
        dst_tri = [stretched_vertices[i] for i in tri]
        rasterize(src_img, dst_img, src_tri, dst_tri)


    return cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)


