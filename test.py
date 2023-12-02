import os
import numpy as np
import pandas as pd
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser

from adet.config import get_cfg
from eval import COCOEvaluator_SIS


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def SIS_evaluation(cfg, task, weight_list, dataset, output_path, inference_dir):
    model = DefaultTrainer.build_model(cfg)
    model.eval()
    torch.set_printoptions(threshold=np.inf)

    with torch.no_grad():
        result = {'iter': [], 'AP': [], 'AP50': [], 'AP70': [], 'APs': [], 'APm': [], 'APl': []}

        for weight in weight_list:
            print(weight)
            DetectionCheckpointer(model).load(weight)

            res = DefaultTrainer.test(cfg, model, evaluators=[
                COCOEvaluator_SIS(dataset, tasks=(task, ), max_dets_per_image=100, output_dir=inference_dir)
            ])
            for k in result.keys():
                if k == 'iter':
                    result['iter'].append(weight.split('_')[-1].split('.')[0])
                else:
                    result[k].append(res[task][k])

    res = pd.DataFrame(result)
    res.to_csv(output_path, index=False)


args = default_argument_parser().parse_args()
register_coco_instances("ILSO1K_trainval", {},
                        'SIS_datasets/ILSO-1K/trainval/trainval.json',
                        'SIS_datasets/ILSO-1K/trainval/imgs/')
register_coco_instances("ILSO1K_test", {},
                        'SIS_datasets/ILSO-1K/test/test.json',
                        'SIS_datasets/ILSO-1K/test/imgs/')
register_coco_instances("ILSO2K_trainval", {},
                        'SIS_datasets/ILSO-2K/trainval/trainval.json',
                        'SIS_datasets/ILSO-2K/trainval/imgs/')
register_coco_instances("ILSO2K_test", {},
                        'SIS_datasets/ILSO-2K/test/test.json',
                        'SIS_datasets/ILSO-2K/test/imgs/')
register_coco_instances("SOC_train", {},
                        'SIS_datasets/SOC/train/train.json',
                        'SIS_datasets/SOC/train/Imgs/')
register_coco_instances("SOC_val", {},
                        'SIS_datasets/SOC/val/val.json',
                        'SIS_datasets/SOC/val/Imgs/')
register_coco_instances("SOC_test", {},
                        'SIS_datasets/SOC/test/test.json',
                        'SIS_datasets/SOC/test/Imgs/')
register_coco_instances("COME_train", {},
                        'SIS_datasets/COME15K/train/train.json',
                        'SIS_datasets/COME15K/train/imgs_right/')
register_coco_instances("COME_e", {},
                        'SIS_datasets/COME15K/test/COME-E/come_e.json',
                        'SIS_datasets/COME15K/test/COME-E/RGB/')
register_coco_instances("COME_h", {},
                        'SIS_datasets/COME15K/test/COME-H/come_h.json',
                        'SIS_datasets/COME15K/test/COME-H/RGB/')
register_coco_instances("SIS10K_train", {},
                        'SIS_datasets/SIS10K/Train/train.json',
                        'SIS_datasets/SIS10K/Train/Image/')
register_coco_instances("SIS10K_val", {},
                        'SIS_datasets/SIS10K/Val/val.json',
                        'SIS_datasets/SIS10K/Val/Image/')
register_coco_instances("SIS10K_test", {},
                        'SIS_datasets/SIS10K/Test/test.json',
                        'SIS_datasets/SIS10K/Test/Image/')

cfg = setup(args)

if 'MODEL.WEIGHTS' in args.opts:
    weight_list = [cfg.MODEL.WEIGHTS]
else:
    weight_list = [cfg.OUTPUT_DIR + '/model_final.pth']

output_path = os.path.join(cfg.OUTPUT_DIR, 'eval_results/' + cfg.DATASETS.TEST[0] + '.csv')

if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'eval_results/')):
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'eval_results/'))

SIS_evaluation(
    cfg,
    task='segm',
    weight_list=weight_list,
    dataset=cfg.DATASETS.TEST[0],
    output_path=output_path,
    inference_dir=cfg.OUTPUT_DIR + '/' + 'inference/'
)