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
bash train_sls_carnX4.sh $gpu $target_budget  # Training on DIV2K
```

For test,
```bash
bash test_sls_carnX4.sh $gpu $exp_name    # Test on Set14, B100, Urban100
```

To see the computational costs (w.r.t MACs and Num. Params.) of a trained model,
```bash
bash compute_costs.sh $gpu $model_dir
```

### Deblurring
The codes will be available soon.


## Acknowledgment
Our implementation is based on the following repositories:
* [Deblurring](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
* [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
* [CARN](https://github.com/nmhkahn/CARN-pytorch.git)
* [Learning N:M Sparsity from Scratch](https://github.com/NM-sparsity/NM-sparsity.git)