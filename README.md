## Codebase for [*Improving Text Style Transfer using Masked Diffusion Language Models with Inference-time Scaling*](https://arxiv.org/abs/2508.10995)

## Setup:
The code is based on PyTorch, `xformers`, and Flash attention. Follow the below steps to setup the conda environment:
```sh 
pip install xformers==0.0.29.post3
```
Installing xformers auto-installs the required PyTorch version as well
```sh 
pip install transformers==4.52.1
```
```sh 
pip install sentence-transformers==4.1.0
```
```sh 
pip install pytorch-lightning==2.0.9
```
```sh 
pip install numpy==1.26.0
```
```sh 
pip install lightning==2.5.1
```
```sh 
pip install flash-attn==2.7.3 --no-build-isolation
```

Clone the [Flash attention github](https://github.com/Dao-AILab/flash-attention) and do the following:
```sh
cd csrc/rotary && pip install .
```
```sh
cd csrc/xentropy && pip install .
```
```sh
cd csrc/layer_norm && pip install .
```
