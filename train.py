from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from adet.config import get_cfg


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

def main(args):
    cfg = setup(args)

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

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
