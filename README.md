
## 1. About the Project <a name="atp"></a>
 **Course:**  Applied Image Processing (CS4365) \
 **Program:** Msc Robotics @ TU Delft            
 **Developer:**    Jakob Bichler    
 **Student ID** 5856795     

This repository is the submission for the final assignment of the course **Applied Image Processing, "Seam-Carved Vectorization"**.



--- 

## Table of Contents

1. [About the Project](#atp) 

2. [Setting up the project in a new virtual environment](#setup)

3. [Usage](#u)
    
4. [File Structure](#fs)

5. [Links to algorithmic steps](#as)


## 2. Setting up the project in a new virtual environment [LINUX]<a name="setup"></a>

Follow these steps to set up the project on a Linux machine in a new Python virtual environment:

### 2.1.  (Optional) Install the `python3.8-venv` package
If you don't have the `python3.8-venv` package installed, you can install it using:

```
sudo apt install python3.8-venv
```
### 2.2. Create a new virtual environment
Navigate to the directory where you'd like to set up your virtual environment and run:

```
python3.8 -m venv aip_env
```

This command will create a new virtual environment named aip_env. You can **replace ``aip_env`` with any name you prefer**.

### 2.3. Activate the virtual environment

```
cd aip_env
```

```
source bin/activate
```
After running this command, your terminal prompt should change to indicate that the virtual environment is active.

### 2.4. Clone the repository
```
git clone git@gitlab.ewi.tudelft.nl:cgv/cs4365/student-repositories/2023-2024/cs436523jbichler.git
```
```
cd cs436523jbichler
```

### 2.5. Install the required packages

With your virtual environment activated, install the necessary packages using:
```
pip install -r requirements.txt
```
Now, your project is set up and ready to run in a fresh virtual environment.

## 3. Usage <a name="u"></a>

To run the `seam_carving.py` script, you can use the following command:

```bash
python3 seam_carving.py --image_path IMAGE_PATH --class_id CLASS_ID --n_cols N_COLS --n_rows N_ROWS --depth_weight DEPTH_WEIGHT [--show_steps]
```

Arguments:

-- image_path: Path to the input image. For example: "data/images/lion.jpg".

 -- class_id: ImageNet class to perform Grad-CAM with. Example value: 291 for lion. All 1000 ImageNet classes are possible. For reference: [List of all ImageNet labels](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)

-- n_cols: Number of vertical seams to remove

-- n_rows: Number of horizontal seams to remove

-- depth_weight: Weight for the depth estimate in the combination of inpainted heatmap from Gradcam, gradient energy and depth estimation

-- show_steps: (Optional) Flag. If provided, the script will show and save intermediate steps. There's no need to provide a value for this flag. It will save intermediate output images and create a video displaying the steps of seam carving



Example:
```
python3 seam_carving.py --image_path "data/images/lion.jpg" --class_id 291 --n_cols 3 --n_rows 3  --depth_weight 0.3 --show_steps
```


The script can run on any ``.jpg`` image that you download to the ``data/images`` directory. I provided some sample images like lion, scorpion and cat_dog 

### 4. File structure <a name="fs"></a>


### 5. Links to algorithmic steps<a name="as"></a>



