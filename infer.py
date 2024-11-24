import argparse
import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
from model import PolypModel

class Nhap:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)['image']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Image Path')
    return parser.parse_args()

def mask2rgb(mask):
    color_dict = {
        0: torch.tensor([0, 0, 0]),
        1: torch.tensor([1, 0, 0]),
        2: torch.tensor([0, 1, 0])
    }
    output = torch.zeros((mask.shape[0], mask.shape[1], 3)).long()
    for k in color_dict.keys():
        output[mask.long() == k] = color_dict[k]
    return output.to(mask.device)

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    return {
        'ids': ids,
        'strings': strings,
    }

def load_and_preprocess_image(path):
    if not os.path.exists(path):
        print(f"Error: Image file {path} does not exist")
        sys.exit(1)
        
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read image {path}")
        sys.exit(1)
    
    H, W = img.shape[:2]
    
    test_transform = Nhap()
    img_tensor = test_transform(img)
    
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, H, W

def main():
    args = parse_args()
    
    try:
        model = PolypModel.load_from_checkpoint('quachtuananh.ckpt')
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        img, H, W = load_and_preprocess_image(args.path)
        img = img.to(device)
        
        output_dir = 'predicted'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with torch.no_grad():
            predicted_mask = model(img)
        
        input_filename = os.path.splitext(os.path.basename(args.path))[0]
        
        argmax = torch.argmax(predicted_mask[0], 0)
        one_hot = mask2rgb(argmax).float().permute(2, 0, 1)
        mask2img = Resize((H, W), interpolation=InterpolationMode.NEAREST)(
            ToPILImage()(one_hot))
        
        output_path = os.path.join(output_dir, f"Predicted_{input_filename}.png")
        mask2img.save(output_path)
 
        print(f"Check {output_path}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()