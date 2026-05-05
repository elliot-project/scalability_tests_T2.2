## Overview

FlagScale https://github.com/flagos-ai/FlagScale/tree/main  is an open source tool that handles generative AI model pretraing (LLM & multimodal) and that can be run on Leonardo on multiple nodes. It is based on different tools such as Megatron-LM. For LLM models pre training from scratch without a checkpoint is supported (Tocheck with multimodal).

Clone the FlagScale repository

`git clone https://github.com/flagos-ai/FlagScale/tree/main`


## Data preprocessing
Download LLava dataset from original repo ad unzip the images folder

`module load git-lfs
git-lfs clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip
`


The dataset now need to be converted in the WebDataset format using the data_preparation.sh script changing the MSC_CONFIG, dataset, vision-root and dataset rooth path.

`sbatch data_preparation.sh`

Please check that the WebDataset is properly created (just need to look at the error file). If this step is ended successfully you should have the following structure in the LLaVA-folder



## Pretraining
Run qwen_vl_run_job.sh setting properly the environment variables and the data & model path.

`sbatch qwen_vl_run_job.sh`

Currently not working for a problem with the tokenizer please look at /leonardo_scratch/large/userinternal/dbrandon/ELLIOT/38869642.err


## License
This project is based on FlagScale project.
This project is licensed under the [Apache License (Version 2.0)](./LICENSE).
This project also contains other third-party components under other open-source licenses.
