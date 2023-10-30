import argparse

import cv2
import numpy as np
import torch

from backbones import get_model
from expression.models import SwinTransFER
from torchvision.transforms import Compose, Normalize, ToTensor

def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

@torch.no_grad()
def inference(weight, name, img_path):

    img = cv2.imread(img_path, 1)[:, :, ::-1]
    img = cv2.resize(img, (112, 112))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])


    swin = get_model(name)


    net = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)

    dict_checkpoint = torch.load(weight)
    net.load_state_dict(dict_checkpoint["state_dict_model"])

    net.eval()
    output, hm = net(input_tensor)
    print(output)
    hm = hm.numpy()
    print(hm.shape)

    #hm = hm[0, 3]
    hm = np.max(hm[0], axis=0)
    print(hm.shape)


    hm = cv2.resize(hm, (112, 112))
    cam_image = show_cam_on_image(img=img, mask=hm, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("test.jpg", cam_image)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='swin_t', help='backbone network')
    parser.add_argument('--weight', type=str, default='/home/pris/qin/swinFER/output/v1/t0/01/checkpoint_step_19999_gpu_0.pt')
    parser.add_argument('--img', type=str, default='/home/pris/qin/FERCAM/code/swinFER_v1/test_0012_aligned.jpg')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
