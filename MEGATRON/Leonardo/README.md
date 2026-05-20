## Creation of a Singularity Container from NVIDIA NeMo 25.02

To create the Singularity container with NeMo 25.02 run the create_container.sh jobscript. Please check that the selected partition is "ldr_all_serial".
```
sbatch create_container.sh
```
## Dataset preparation
### Pretraining

1. Download the LLavA-Pretrain dataset from Hugging Face and unzip the images folder (NOTE: 79GB of disk space required):

    ```
    git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
    cd LLaVA-Pretrain
    unzip images.zip
    ```

3. Run the following script to convert the data to webdataset format:

    ```
    cd <megatron-lm dir>
    python examples/multimodal/convert_llava_pretrain_to_wds.py
    ```

4. Run the following command to convert to megatron-energon format:

    ```
    cd <LLaVA-Pretrain dir>/wds
    energon prepare ./
    ```

    select the following values for the presented options:

    ```
    > Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
    > Do you want to create a dataset.yaml interactively? [Y/n]: Y
    > Please enter a number to choose a class: 9 (VQASample)
    > Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
    > Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
    > Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
    > Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
    > Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
    ```

## Configuration
Edit `train_qwen2_5-vl_llava.sh` scripts with the paths of the container, data and output folders

## Training
launch qwenvl training:
 ```
 sbatch train_qwen2_5-vl_llava.sh
 ```
