# SLS-CVPR2022
#### Junghun Oh, Heewon Kim, Seungjun Nah, Cheeun Hong, Jonghyun Choi, Kyoung Mu Lee
This repository is a Pytorch implementation of the paper **"Attentive Fine-Grained Structured Sparsity for Image Restoration"** from **CVPR2022**. [[arXiv](https://arxiv.org/abs/2204.12266)]

If you find this code useful for your research, please consider citing our paper:
```
@InProceedings{Oh_2022_CVPR,
  author = {Oh, Junghun and Kim, Heewon and Nah, Seungjun and Hong, Cheeun and Choi, Jonghyun and Lee, Kyoung Mu},
  title = {Attentive Fine-Grained Structured Sparsity for Image Restoration},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

## Proposed Method
![](/figs/method.png)

## Results
**Quantitative results**
![](/figs/quantitative.png)
**Qualitative results**
![](/figs/qualitative.png)

## Dataset
### Super-Resolution
For super-resolution, we use [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train and validate a model.
You can download it [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)

After training, we evaluate trained models with benchmark datasets ([Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests), [B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), and [Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr)).
You can download them [here](https://cv.snu.ac.kr/research/EDSR/benchmark.tar).

Unpack the downloaded tar files and change the ```args.dir_data``` in ```super-resolution/src/option.py``` to the directory where the DIV2K and benchmark datasets are located.

### Deblurring
For deblurring, we use [GOPRO](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf) dataset to train and evaluate a model.
You can download it [here](https://seungjunnah.github.io/Datasets/gopro).

Unpack the downloaded file and change the ```args.data_root``` in ```deblur/src/option.py``` to the directory where the GOPRO dataset is located.

## Usage
Clone this repository.
```bash
git clone https://github.com/JungHunOh/SLS_CVPR2022.git
cd SLS_CVPR2022
```

### Super-Resolution
```bash
cd super-resolution/src
```

For training,
```bash
bash ./scripts/train_sls_carnX4.sh $gpu $target_budget  # Training on DIV2K
```

For test,
```bash
bash ./scripts/test_sls_carnX4.sh $gpu $exp_name    # Test on Set14, B100, Urban100
```

To see the computational costs (w.r.t MACs and Num. Params.) of a trained model,
```bash
bash ./scripts/compute_costs.sh $gpu $model_dir
```

### Deblurring
The codes will be available soon.


## Acknowledgment
Our implementation is based on the following repositories:
* [Deblurring](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
* [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
* [CARN](https://github.com/nmhkahn/CARN-pytorch.git)
* [Learning N:M Sparsity from Scratch](https://github.com/NM-sparsity/NM-sparsity.git)