import cv2
import numpy as np
from pygetwindow import Rect
from utils import *
    
FCG = SimulatorApp('FCG', Rect(10, 40, -10, -10))

det_culane_cfg = {
    "griding_num": 200,
    "num_lanes": 4,
    "test_model": r"D:\AI_LaneDetection_Hoang_HVCH\Game_FCG\NavigationService\models\ufsd\culane_18.pth",
    "cls_num_per_lane": 18,
    "backbone": "18"
}
detector = LaneDetector(det_culane_cfg)

while True:
    # get frame    
    frame = FCG.capture_frame() 
    
    # detect lane
    frame = detector.detect(frame)
    
    # navigation with lane

    # show image
    cv2.imshow("test", frame)
    
    # loop
    cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break