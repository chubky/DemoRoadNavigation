import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch
from evaluation.eval_wrapper import generate_lines
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import scipy.special
import cv2
from data.constant import culane_row_anchor, tusimple_row_anchor

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),
                    use_aux=False).cpu() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)
    print(type(net))
    
    net.eval()
    if cfg.dataset == 'CULane':
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        output_path = r'D:\AI_LaneDetection_Hoang_HVCH\Ultra-Fast-Lane-Detection\inferrences'
        # Load the image
        image = Image.open(r'D:\AI_LaneDetection_Hoang_HVCH\Ultra-Fast-Lane-Detection\inferrences\00000.jpg')
        
        imgs, names = img_transforms(image).unsqueeze(0), "test"
        imgs = imgs.cpu()
        with torch.no_grad():
            out = net(imgs)
        if len(out) == 2:
            out, seg_out = out

        generate_lines(out, imgs[0,0].shape, names, output_path, cfg.griding_num, localization_type = 'rel',flip_updown = True)
        
        # visualize
        img_w, img_h = 1640, 590
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter('demo' + 'avi', fourcc , 30.0, (img_w, img_h))
        
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        vis = cv2.imread(r'D:\AI_LaneDetection_Hoang_HVCH\Ultra-Fast-Lane-Detection\inferrences\00000.jpg')
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (culane_row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
        window_name = 'image'
        cv2.imshow(window_name, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()