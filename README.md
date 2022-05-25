# Feature Space Particle Inference for Neural Network Ensembles (ICML2022)
Official Implementation in JAX and Flax.  
arXiv: WIP

## Training and Evaluation
To train an ensemble of 10 WRN-16-4 using feature-WGD, run
```
python train.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature-wgd --dataset cifar100  \
--batch_size 128 --output_dir results
```
To evaluate the above trained model, run
```
python eval.py --num_particles 10 --model_name WideResnet16-4 \
--num_epochs 300 --method feature-wgd --dataset cifar100  \
--batch_size 128 --output_dir results
```

Training curves and evaluation results can be loaded using TensorBoard. TensorBoard events will be saved in `output_dir`, and their path will contain the configurations of the experiment.


## Citation
WIP

## Acknowledgement
The code is based on the official implementation of [SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization](https://github.com/google-research/sam).