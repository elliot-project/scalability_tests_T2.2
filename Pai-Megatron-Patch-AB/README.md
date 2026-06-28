# Qwen3-VL Training with Pai-Megatron-Patch

This document describes how to create the container, prepare the dataset, and launch training with `run_qwen.sh`.

## 1. Create the container

Set the Apptainer cache and temporary directories to your CINECA scratch area, then build the image:

```bash
export APPTAINER_CACHEDIR=$CINECA_SCRATCH
export APPTAINER_TMP=$CINECA_SCRATCH
apptainer build qwen3vl_pai_megatron.sif qwen3vl_pai_megatron.def
```

## 2. Create the dataset

Open the container with the binds needed for your output data folder:

```bash
apptainer shell -B $BINDS qwen3vl_pai_megatron.sif
```

Inside the container, build the fake webdataset and then generate the metadata:

```bash
cd /mnt/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_fake_wds_for_vl.py --output-dir your_wds_path

cd /mnt/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_wds_meta_data_from_datajuice.py --dataset-root your_wds_path
```

## 3. Launch training

Before launching training, edit `run_qwen.sh` and change all paths at the beginning of the script to match your environment.

Then start training with:

```bash
bash run_qwen.sh
```

The log output should be similar to the one reported in the official GitHub repository [web:6][web:11].

## Official repository

For any further information, please refer to the official repository:

[https://github.com/alibaba/Pai-Megatron-Patch.git](https://github.com/alibaba/Pai-Megatron-Patch.git)

## Caveat

The code is currently not working with other datasets.
