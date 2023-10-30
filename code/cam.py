import argparse

import cv2
import numpy as np
import torch
import os

from backbones import get_model
from expression.models import SwinTransFER
from torchvision.transforms import Compose, Normalize, ToTensor

from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@torch.no_grad()
def inference(weight, name, image_path, method="scorecam"):

    swin = get_model(name)
    model = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True).cuda()
    dict_checkpoint = torch.load(weight)
    model.load_state_dict(dict_checkpoint["state_dict_model"])
    model.eval()

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    
    target_layers = [model.norm]
    targets=[ClassifierOutputTarget(i) for i in range(7)]


    if method == "ablationcam":
        cam = methods[method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=True,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=True,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (112, 112))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=False,
                        aug_smooth=False)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{method}_cam.jpg', cam_image)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='swin_t', help='backbone network')
    parser.add_argument('--weight', type=str, default='/home/pris/qin/swinFER/output/v1/t0/01/checkpoint_step_19999_gpu_0.pt')
    parser.add_argument('--img', type=str, default='/home/pris/qin/FERCAM/code/swinFER_v1/test_0012_aligned.jpg')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
