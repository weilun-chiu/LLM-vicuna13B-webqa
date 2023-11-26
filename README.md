# Vicuna-13B-WebQA-LLM

Welcome to the Vicuna-13B deployment with [GenRead](https://arxiv.org/abs/2209.10063) on the [WebQA](https://webqna.github.io/) dataset. Unfortunately, the evaluation script has not been released by [WebQA](https://webqna.github.io/), making it challenging to assess performance.

Please note that the primary repository for this project is not public due to CMU's policy. However, this repository contains a fork of partial data that does not fall under those restrictions.

## Installation
1. Download the dataset from [WebQA](https://webqna.github.io/).
2. Follow the Model Weights section from [FastChat](https://github.com/lm-sys/FastChat).
3. Install the required packages from the ongoing development in the `requirements.txt` file.

## Usage
Run the following command to initiate the process:
```bash
./run_vicuna.sh
```

## License
Please follow [Meta's LLaMA model license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
