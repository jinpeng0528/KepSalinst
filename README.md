# [IEEE TCyb] KepSalinst: Using Peripheral Points to Delineate Salient Instances

This is an official implementation of the paper "KepSalinst: Using Peripheral Points to Delineate Salient Instances", accepted by IEEE Transactions on Cybernetics.
[[paper]](https://ieeexplore.ieee.org/abstract/document/10314036)

## Installation
This repository has been tested with the following environment:
* CUDA (11.3)
* Python (3.9.12)
* Pytorch (1.10.1)
* Detectron2 (0.6+cu113)
* Pandas (1.5.2)
* Scipy (1.10.0rc1)

### Example conda environment setup
```bash
conda create -n kepsalinst python=3.9.12
conda activate kepsalinst
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# under your working directory
git clone https://github.com/MonkeyKing0528/KepSalinst.git
cd KepSalinst
python setup.py build develop
pip install -r requirements.txt
```

## Getting Started

### Datasets
To meet Detectron2's specifications, our project employs COCO-style annotated datasets. There are two ways to acquire these: Firstly, download the raw datasets from official sources and manually create COCO-style annotations. Alternatively, for ease, we offer pre-annotated datasets in COCO format, available for download from [OneDrive](https://1drv.ms/f/c/7be8ecfc440137f7/Eggvtb4P1FNLgw47hpu7168Bmr8bFly70xdZKPBYnRtJWQ?e=gbxbIF) or [Baidu Netdisk](https://pan.baidu.com/s/1J6cjC91Q6Fzbn3kOt5GzYA) (code: `wub4`). Notably, the ILSO-1K dataset annotations are sourced from [RDPNet](https://github.com/yuhuan-wu/RDPNet/tree/master), while we generated annotations for the remaining datasets ourselves.

After acquiring the datasets, place them in the `./SIS_datasets/` directory. Alternatively, you can create a symbolic link from the directory containing the datasets to "./SIS_datasets/".

## Training
To train our model, run:
```bash
python train.py --config-file configs/KepSalinst/{dataset}.yaml
```
Replace `{dataset}` with the dataset name, choosing from `ILSO1K`, `ILSO2K`, `SOC`, `COME15K`, or `SIS10K`.

## Evaluation
To evaluate after training, run:
```bash
python test.py --config-file configs/KepSalinst/{dataset}.yaml
```
Or, download our trained models from [OneDrive](https://1drv.ms/f/c/7be8ecfc440137f7/EnP0CJ3lR4VOhz-WhZyToeMBP6WKP868NjbWjVSSDpC55A?e=RFIZpk) or [Baidu Netdisk](https://pan.baidu.com/s/1OCjs_AxFfvNcN0C0gZUMOg) (code: `swbr`). Then, use the following command for evaluation:
```bash
python test.py --config-file configs/KepSalinst/{dataset}.yaml MODEL.WEIGHTS {path to model}
```

## Scripts
To facilitate ease of use, you can directly run the `.sh` files located in the `./scripts/` directory. These files offer complete commands for training and evaluating on different datasets.

## Citation

```
@article{chen2023kepsalinst,
  title={{KepSalinst}: Using Peripheral Points to Delineate Salient Instances},
  author={Chen, Jinpeng and Cong, Runmin and Ip, Horace Ho Shing and Kwong, Sam},
  journal={IEEE Transactions on Cybernetics},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
* This code is based on the CondInst ([2020-ECCV] Conditional convolutions for instance segmentation) implementation from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).
* The ILSO-1K dataset annotations in our work are sourced from [RDPNet](https://github.com/yuhuan-wu/RDPNet/tree/master)
