# Feature Space Particle Inference for Neural Network Ensembles (ICML2022)
Official Implementation in JAX and Flax.  
arXiv: WIP

## Installation
Requirements:
- Linux (Ubuntu 16.04 or later) 
- Python ≥ 3.9
- CUDA ≥ 11.1 
- cuDNN ≥ 8.2 
```
pip install -r requirements.txt
```

## Training and Evaluation
This repository provides training and evaluation of ensemble image classification model using feature-WGD and Deep Ensembles.

To train an ensemble of 10 WRN-16-4 on CIFAR-100 using feature-WGD, run
```
python train.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature_wgd --dataset cifar100  \
--batch_size 128 --output_dir results
```
To evaluate the above trained model, run
```
python eval.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature_wgd --dataset cifar100  \
--batch_size 128 --output_dir results
```

Training curves and evaluation results can be loaded using TensorBoard. TensorBoard events will be saved in `output_dir`, and their path will contain the configurations of the experiment.

For CIFAR-10 and CIFAR-100, dataset is automatically downloaded in `data_dir` using Tensorflow Datasets. For ImageNet, manual download is required (see [instructions](https://www.tensorflow.org/datasets/catalog/imagenet2012)).

To see a detailed list of all available flags, run `python train.py --help`.

## Citation
```
@inproceedings{yashima2022feature,
  author    = {Shingo Yashima, Teppei Suzuki, Kohta Ishikawa, Ikuro Sato and Rei Kawakami},
  title     = {Feature Space Particle Inference for Neural Network Ensembles},
  booktitle = {International Conference on Machine Learning},
  publisher = {PMLR},
  year      = {2022},
}
```

## Acknowledgement
The code is based on the official implementation of [SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization](https://github.com/google-research/sam).