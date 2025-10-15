import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models_sam2 import sam_semi_stochasticdepth_LoraAdapterexternal as sam_model
from semi_transform.semi import SemiDataset
import cv2


def draw_mask(image, mask_generated) :
    masked_image = image.copy()

    masked_image = np.where(mask_generated.astype(int),
                          np.array([0, 0, 255], dtype='uint8'),
                          masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.7, masked_image, 0.3, 0)


def eval(eval_loader, model, threshold, device):
    model.eval()

    pbar = tqdm(total=len(eval_loader), leave=True, desc='val')
    TP, TP_and_FP, TP_and_FN, TN = 0, 0, 0, 0
    with torch.no_grad():
        for inp, gt, mask_name in eval_loader:
            inp = inp.to(device)
            gt = np.array(gt.squeeze().detach().cpu()).astype(np.int_)
            gt[gt > 0] = 1
            pred = model(inp)

            current_TP_and_FN = gt.sum(0).sum(0)

            pred = transforms.functional.resize(pred, gt.shape, transforms.InterpolationMode.BILINEAR)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze().detach().cpu().numpy()
            pred = np.where(pred > threshold, 1, 0)

            if args.save:
                image = cv2.imread(os.path.join(args.test_path, 'JPEGImages', mask_name[0].replace('.png', '.jpg')))
                mask = np.array([pred for i in range(3)]).transpose(1, 2, 0)
                cv2.imwrite(os.path.join(args.save_path, mask_name[0].replace('.png', '.jpg')), draw_mask(image, mask))

            current_TP_and_FP = pred.sum(0).sum(0)
            current_TP = (gt * pred).sum(0).sum(0)
            current_TN = np.where((gt + pred) > 0, 0, 1).sum(0).sum(0)

            TP += current_TP
            TP_and_FP += current_TP_and_FP
            TP_and_FN += current_TP_and_FN
            TN += current_TN
            if pbar is not None:
                pbar.update(1)

    if TP_and_FP == 0:
        precision = 0
    else:
        precision = TP / TP_and_FP
    recall = TP / TP_and_FN
    specificity = TN / (TN + TP_and_FP - TP)
    IoU = TP / (TP_and_FP + TP_and_FN - TP)
    Dice = 2 * TP / (TP_and_FP + TP_and_FN)
    G_Mean = (recall * specificity) ** 0.5

    if pbar is not None:
        pbar.close()

    return precision, recall, specificity, IoU, Dice, G_Mean


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', default='./test_set', help='path to test images and groundtruth')
parser.add_argument('--vision_config', default='./configs/sam2_configs/cod-sam-vit-s-semi_30epoch.yaml', help='path to sam2 model configs')
parser.add_argument('--vision_checkpoint', default='./checkpoints/uwassess_vision.pth', help='path to adapted sam2 model')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_path', default='./results/visual_perception', help='path to save results')
parser.add_argument('--save', default=False, help='whether to save visual perception results')
args = parser.parse_args()

global log_info
device = torch.device(args.device)
with open(args.vision_config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model_args = config['model']['args']

val_data = SemiDataset(root=args.test_path, mode='val', size=model_args['inp_size'])
val_loader = DataLoader(dataset=val_data, batch_size=1, num_workers=8)

model = sam_model.SAM(inp_size=model_args['inp_size'], encoder_mode=model_args['encoder_mode'])
model = model.to(device)

checkpoint = torch.load(args.vision_checkpoint, map_location=torch.device(args.device))
model.load_state_dict(checkpoint)

threshold = 0.5
log_info = []
result1, result2, result3, result4, result5, result6 = eval(val_loader, model, threshold, device)
metric1, metric2, metric3, metric4, metric5, metric6 = 'precision', 'recall', 'specificity', 'IoU', 'Dice', 'G_Mean'
log_info.append('val: {}={:.4f}'.format(metric1, result1))
log_info.append('val: {}={:.4f}'.format(metric2, result2))
log_info.append('val: {}={:.4f}'.format(metric3, result3))
log_info.append('val: {}={:.4f}'.format(metric4, result4))
log_info.append('val: {}={:.4f}'.format(metric5, result5))
log_info.append('val: {}={:.4f}'.format(metric6, result6))
for performance in log_info:
    print(performance)
