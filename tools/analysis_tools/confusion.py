# reference: https://github.com/jacobgil/pytorch-grad-cam
from __future__ import division
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import argparse
import importlib
import os
import os.path as osp
import time
import mmcv
from mmcv.runner import load_checkpoint
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from openbioseq import __version__
from openbioseq.apis import set_random_seed, train_model
from openbioseq.datasets import build_dataset
from openbioseq.datasets import ProcessedDataset
from openbioseq.models import build_model
from openbioseq.utils import (collect_env, get_root_logger, traverse_replace,
                             setup_multi_processes)
try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')


from sklearn.metrics import confusion_matrix
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == '__main__':
    """ Usage of grad-CAM visualization
        version 01.05

    Settings:
        class_num: set the class num of your dataset
        data_name: choose a dataset
        model_arch: CNN arth such as "resnet50" in torchvision
        model_name: name to save
        model_path: path of linear_classification checkpoint
        img_path: a dict of img_name (string) and img_class (int)
        img_root: dirs of image dataset
    
    Running:
        1. set a config and data_name.
        2. python tools/gradcam.py
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)
    model = build_model(cfg.model)
    cam_pool = []
    element = 'vowel'
    path = '/usr/data/OpenBioSeq/work_dirs/benchmarks/classification/neuro/EEG_speech/cnn/cnn5/zyy/modeCrossEntropyLoss/'
    cfg_list = os.listdir(path)
    for _dir in cfg_list:
        if(element in _dir):
            seed = _dir.split('d')[-1]
            try:
                cfg_path = os.path.join(path, _dir)
                model_path = os.path.join(cfg_path, 'epoch_200.pth')
                load_checkpoint(model, model_path, map_location='cpu')
                model.eval()
                model.zero_grad()
                cfg.data.train.data_source.seed = int(seed)
                datasets = [build_dataset(cfg.data.train)]
                data = datasets[0].data_source.data['test']
                targets = datasets[0].data_source.targets['test']
                input = data.requires_grad_(True)
                x = input
                for name, module in model._modules.items():
                    x = module(x)
                output = x[-1].detach().numpy()
                output = np.argmax(output,axis=-1)
                if(element=='initial'):
                    targets = targets[:,-1]
                index = targets
                # Create a confusion matrix
                cm = confusion_matrix(targets.cpu().detach().numpy(), output)
                cam_pool.append(cm)
                # print(cam)
            except BaseException as e:
                print('Failed to do something: ' + str(e))

    # fit_model_state_dict(model, "/usr/data/OpenBioSeq/work_dirs/benchmarks/classification/neuro/EEG_speech/cnn/cnn5/htp_tone/epoch_50.pth", map_location='cpu')
    averaged_cam = np.sum(cam_pool, axis=0)
    # averaged_cam = np.mean()np.array(cam_pool)
    print(averaged_cam)
    print(averaged_cam.shape)
    true_positives = np.diag(averaged_cam).sum()
    
    # Calculate the total number of samples
    total_samples = averaged_cam.sum()
    
    # Calculate accuracy
    accuracy = true_positives / total_samples
    print(accuracy)
    df = pd.DataFrame(averaged_cam) #convert to a dataframe
    df.to_csv(os.path.join(path, element + "_averaged__cm.csv"),index=False) #save to file
    print(averaged_cam)
    # default_traget_layers = get_default_traget_layers(model)
    # print(default_traget_layers)
    # model.to(args.device)

    # cam = init_cam(model, default_traget_layers, True, reshape_transform=None)

    

    # data = datasets[0].data_source.data['test']
    # targets = datasets[0].data_source.targets['test']
    # input = data.requires_grad_(True)
    # x = input
    # for name, module in model._modules.items():
    #     x = module(x)
    # output = x[-1].cuda()
    # index = targets
    # one_hot = np.zeros((targets.shape[0], output.size()[-1]), dtype=np.float32)
    # one_hot[:, index] = 1
    # one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()
    # one_hot = torch.sum(one_hot.cuda() * output)
    # one_hot.backward(retain_graph=True)
    # model.zero_grad()
    # cam = torch.sqrt(torch.mean(input.grad**2, dim=0))
    # cam = torch.mean(cam, dim=1)
    # cam = cam - torch.min(cam)
    # cam = cam / torch.max(cam)
    # cam = cam.numpy()
    # df = pd.DataFrame(cam) #convert to a dataframe
    # df.to_csv("/usr/data/OpenBioSeq/work_dirs/benchmarks/classification/neuro/EEG_speech/cnn/cnn5/htp/modeCrossEntropyLoss/tone_saliency.csv",index=False) #save to file
    # print(cam)
    # grayscale_cam = cam(data.unsqueeze(0), targets)
                        
        # eigen_smooth=args.eigen_smooth,
        # aug_smooth=args.aug_smooth
    # default_traget_layers = ['0']
    # grad_cam = GradCam(model=model, feature_module=model.backbone.layer4, target_layer_names=default_traget_layers, use_cuda=torch.cuda.is_available())
    # datasets = [build_dataset(cfg.data.train)]
    # data = datasets[0].data_source.data['test'][0]
    # # print(datasets[0].data_source.targets['test'].shape)
    # input = data.requires_grad_(True)
    # print(input.shape)
    # # print(input.shape)
    # target_index = None
    # mask = grad_cam(input.unsqueeze(0), target_index)
    # print(mask.shape)


    # args = {
    #     # ----- dataset 1: cub200 ------
    #     "class_num": 200,
    #     "data_name": "cub200",
    #     # ------ dataset 2: pets -------
    #     # "class_num": 37,
    #     # "data_name": "pets37",
    #     # ----- dataset 3: dogs120 -----
    #     # "class_num": 120,
    #     # "data_name": "dogs120",
    #     # -------- cnn arch ----------
    #     "model_arch": "resnet50",
    #     # ------------------------- Example: SL imagenet path -----------------------------
    #     "model_name": "imagenet_r50_pytorch",  # model name 1
    #     "model_path": "./work_dirs/benchmarks/linear_classification/cub200/r50_last_2gpu_cub200/imagenet_r50_pytorch.pth/epoch_100.pth",  # cub200 baseline
    #     # "model_path": "./work_dirs/benchmarks/linear_classification/pets/r50_last_2gpu_pets/imagenet_r50_pytorch.pth/epoch_100.pth",  # Pets baseline
    #     # "model_path": "./work_dirs/benchmarks/linear_classification/dogs120/r50_last_2gpu_dogs120/imagenet_r50_pytorch.pth/epoch_100.pth",  # Dogs baseline
    #     # "model_name": "imagenet_r50_pytorch_bbox",  # model name 2
    #     # "model_path": "./work_dirs/benchmarks/linear_classification/cub200/r50_last_2gpu_cub200_ablation/imagenet_r50_pytorch.pth/epoch_100.pth",  # cub200, no back: bbox
    #     # "model_name": "imagenet_r50_pytorch_mask",  # model name 3
    #     # "model_path": "./work_dirs/benchmarks/linear_classification/pets/r50_last_2gpu_pets_ablation/imagenet_r50_pytorch.pth/epoch_100.pth",  # pets, no back: mask
    #     # -------- image dict ----------
    #     "img_path": dict(),
    #     # ------- dataset root ---------
    #     # "img_root": "/usr/commondata/public/CUB200/CUB_200/images/",  # CUB-200, ori img
    #     # "img_root": "/usr/commondata/public/CUB200/CUB_200/images_bbox/",  # CUB-200, no back: bbox
    #     # "img_root": "/usr/commondata/public/Pets37/images/",  # Pets-37, ori img
    #     "img_root": "/usr/commondata/public/Pets37/images_segmented/",  # Pets-37, no back: mask 
    #     # "img_root": "/usr/commondata/public/Dogs120/Images/",  # Dogs-120, ori img
    #     "use_cuda": torch.cuda.is_available()
    # }
    # assert args["data_name"] in ["cub200", "pets37", "dogs120"]

    # # get sample dict
    # try:
    #     args["img_path"] = eval("get_sample_{}_dict()".format(args["data_name"]))
    # except:
    #     print("Please add 'get_sample_{}_dict()' to load img list!".format(args["data_name"]))
    #     exit()
    
    # i = 0
    # for k,v in args["img_path"].items():

    #     cnn_arch = models.__dict__[args["model_arch"]]
    #     model = cnn_arch(num_classes=args["class_num"])


        
    #     fit_model_state_dict(model, args["model_path"])
    #     grad_cam = GradCam(
    #         model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=torch.cuda.is_available())
    #     print(i)

    #     if len(args["img_root"]) > 1:
    #         img_path = args["img_root"] + k
    #     else:
    #         img_path = k
        
    #     img = cv2.imread(img_path, 1)
    #     img = np.float32(cv2.resize(img, (224, 224))) / 255
    #     input = preprocess_image(img)

    #     # save to plot dir
    #     save_path = "work_dirs/plot_cam/{}/{}".format(args["data_name"], args["model_name"])
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     # choose a target label, set None as default
    #     target_index = None
    #     mask = grad_cam(input, target_index)

    #     gb_model = GuidedBackpropReLUModel(model=model, use_cuda=torch.cuda.is_available())

    #     gb, cls_res = gb_model(input, index=target_index, target=v)
    #     gb = gb.transpose((1, 2, 0))
    #     cam_mask = cv2.merge([mask, mask, mask])
    #     cam_gb = deprocess_image(cam_mask * gb)
    #     gb = deprocess_image(gb)

    #     cls_res_str = str(cls_res["bool"]) if cls_res["bool"] else "{}_pred{}".format(str(cls_res["bool"]), str(cls_res["pred"]))
    #     show_cam_on_image(img, mask, "{}_C{}_id{}_{}".format(args["model_name"], str(v), str(i), cls_res_str), save_path)
    #     # cv2.imwrite('{}/{}_id{}_gb.jpg'.format(save_path, args["model_name"], str(i)), gb)
    #     cv2.imwrite('{}/{}_C{}_id{}_{}_cam_gb.jpg'.format(save_path, args["model_name"], str(v), str(i), cls_res_str), cam_gb)
    #     i += 1