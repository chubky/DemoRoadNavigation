
from mss import mss
import pyautogui
from pygetwindow import Rect
import numpy as np
from models.ufsd.model.model import parsingNet
import torch
import torchvision.transforms as transforms
import cv2
import scipy.special
from models.ufsd.data.constant import culane_row_anchor
from models.ufsd.utils.config import Config
from PIL import Image
import time

class SimulatorApp:
    def __init__(self, title: str, margin: Rect) -> None:
        self.window = pyautogui.getWindowsWithTitle(title)[0]
        self.window.resizeTo(1640, 590 + margin.top)
        self.margin: Rect = margin
        self.sct = mss()
        
    def remove_margin(self, frame: Rect) -> Rect:
        return Rect(frame.left + self.margin.left, 
                    frame.top + self.margin.top, 
                    frame.right + self.margin.right, 
                    frame.bottom + self.margin.bottom)

    def capture_frame(self) -> np.ndarray:   
        rbbox = self.window._getWindowRect()         
        bbox = self.remove_margin(rbbox)
        captured = self.sct.grab(bbox)    
        frame = np.array(captured) 
        return frame

class LaneDetector:
    def __init__(self, cfg: dict) -> None:
        self.cfg = Config(cfg)
        self.net = parsingNet(pretrained = False, 
                        backbone=self.cfg.backbone,
                        cls_dim = (self.cfg.griding_num + 1, self.cfg.cls_num_per_lane, self.cfg.num_lanes),
                        use_aux=False).cpu()
        self.load_dict(self.cfg.test_model)
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def load_dict(self, pretrained_path):
        state_dict = torch.load(pretrained_path, map_location = 'cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict = False)
        
    def detect(self, image: np.ndarray) -> np.ndarray:
        #  convert to PIL
        pil_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img) 
        
        # do transform
        imgs = self.img_transforms(pil_img).unsqueeze(0).cpu()
        with torch.no_grad():
            out = self.net(imgs)
            
        if len(out) == 2:
            out, seg_out = out

        # visualize
        img_w, img_h = 1640, 590
        
        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.cfg.griding_num] = 0
        out_j = loc

        # show line
        temp = image.copy()
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, 
                               int(img_h * (culane_row_anchor[self.cfg.cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(temp,ppp,5,(0,255,0),-1)
        cv2.imwrite(f'out/{time.time()}.jpg', temp)
        return temp