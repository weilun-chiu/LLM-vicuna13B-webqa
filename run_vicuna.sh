#!/bin/sh
module load <your module>
conda activate <your conda path>
export HF_HOME=<cache location>
python src/vicuna.py
