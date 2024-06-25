# mmSAM-MoE
This repository contains the code for **mmSAM-MoE: A Multimodal Medical Segment Anything Model with Mixture-of-Experts Segmentation**

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=mmSAMmoe_env.yaml
conda activate mmSAMmoe
```


## Example Usage for Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_modality.py 
```
