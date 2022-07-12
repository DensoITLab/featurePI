# Feature Space Particle Inference for Neural Network Ensembles (ICML2022)
This repository contains experiments for the paper *[Feature Space Particle Inference for Neural Network Ensembles](https://arxiv.org/abs/2206.00944)* by Shingo Yashima, Teppei Suzuki, Kohta Ishikawa, Ikuro Sato, and Rei Kawakami.

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
This repository provides training and evaluation of ensemble image classification model using feature-WGD and Deep Ensembles on CIFAR-10, CIFAR-100, and ImageNet.

To train an ensemble of 10 WRN-16-4 on CIFAR-100 using feature-WGD, run
```bash
python train.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature_wgd --dataset cifar100  \
--batch_size 128 --output_dir <OUTPUT_DIR> --data_dir <DATA_DIR>
```
To evaluate the above trained model, run
```bash
python eval.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature_wgd --dataset cifar100  \
--batch_size 128 --output_dir <OUTPUT_DIR> --data_dir <DATA_DIR>
```

Training curves and evaluation results can be loaded using TensorBoard. TensorBoard events will be saved in `OUTPUT_DIR`, and their path will contain the configurations of the experiment.

For CIFAR-10 and CIFAR-100, dataset is automatically downloaded in `DATA_DIR` using Tensorflow Datasets. For ImageNet, manual download is required (see [instructions](https://www.tensorflow.org/datasets/catalog/imagenet2012)).

To see a detailed list of all available flags, run `python train.py --help`.

## Citation
```BibTeX
@inproceedings{yashima2022feature,
  author    = {Yashima, Shingo and Suzuki, Teppei and Ishikawa, Kohta and Sato, Ikuro and Kawakami, Rei},
  title     = {Feature Space Particle Inference for Neural Network Ensembles},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages     = {25452--25468},
  publisher = {PMLR},
  year      = {2022},
}
```

## Acknowledgement
The code is based on the official implementation of [SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization](https://github.com/google-research/sam).

## License

Our original license. Please see [LICENSE](./LICENSE).
