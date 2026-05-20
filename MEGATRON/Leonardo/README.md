1. Creation of a Singularity Container from NVIDIA NeMo 25.02

To create the Singularity container with NeMo 25.02 run the create_container.sh jobscript. Please check that the selected partition is "ldr_all_serial".
2. Configuration
  Edit train_qwen2_5-vl.sh scripts with the paths of the container, data and output folders

3. Training
   launch qwenvl training:
   ```
   sbatch train_qwen2_5-vl.sh
   ```
