To install the dependencies from the Flagscale README (https://github.com/flagos-ai/FlagScale), use the following commands:
```
git clone https://github.com/flagos-ai/Megatron-LM-FLcd Megatron-LM-FL
pip install .[mlm]

git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
Apex may cause errors with gradient accumulation, so use the --no-gradient-accumulation-fusion flag in the torchrun command.

Following the example README in the train section (FlagScale/docs/getting-started.md), the flagscale train command generates a bash script containing the torchrun command. This command must then be placed into a job script that you write, modifying parameters such as the IP address, port, number of nodes, and number of processes.
