import cv2
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from icecream import ic


class MidasDepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.midas.eval()

    def preprocess_image(self,image_path):
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.original_shape = img.shape[:2]

        # Resize
        img = cv2.resize(img, (384, 384))
        
        # Normalize to [0, 1]
        img = img.astype("float32") / 255.0
        
        # Convert to torch tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) -> (B, C, H, W)
        
        return img, img_tensor

    def estimate_depth(self, image_path):
        img, input_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            prediction = self.midas(input_tensor.to(self.device))

            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Rescale output to original size but flip axes for visualization purposes
        output = cv2.resize(output, self.original_shape[::-1])


        return output