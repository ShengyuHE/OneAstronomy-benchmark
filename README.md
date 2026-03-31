# OneAstronomy Benchmark (currenlty AION) Quick Start Guide

## 1. Data Preparation

- **Download Data**: Please obtain the dataset from the following address
- **Google Drive Link**: <https://drive.google.com/file/d/1kxW4odI3EqIojyCttJ2S1MEZV7gu7GPJ/view?usp=drive_link>
- Extract the downloaded file and place it in the `data` directory of the project.

## 2. Environment Setup

### 2.1 Container Installation (Recommended)

- **Pull Image**: `docker pull zwgcv/aion:v1.0`
- **Create Container**: `docker run -dit --gpus all --name aion_container -v /home/:/mnt zwgcv/aion:v1.0`
- **Enter Container**: `docker exec -it aion_container_id /bin/bash` (replace `aion_container_id` with your container ID)
- **View Container ID**: Use the `docker ps -a` command
- **Activate Environment**: `conda activate aion`

### 2.2 Direct Installation

If you already have a PyTorch environment, you can install AION directly with the following command:

- **Install AION**: `pip install polymathic-aion`

## 3. Model Prediction

- **Run Redshift Prediction**: `python RedshiftPrediction.py`

## 4. Model Training

- **Start Training**: `CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 train_aion_ddp_token.py > training.log 2>&1 & disown`
  - This command will run the training process in the background using 2 GPUs, with output redirected to the training.log file

## 5. Project Structure

```
AION/
├── aion/                # Core source code directory
│   ├── codecs/          # Encoding/decoding related code
│   ├── fourm/           # Specific functionality modules
│   ├── __init__.py
│   ├── modalities.py    # Modality definitions
│   └── model.py         # Model definition
├── data/                # Data directory
│   ├── test/            # Test data
│   ├── train/           # Training data
│   └── best_model.pt    # Test model
├── docs/                # Documentation directory
├── notebooks/           # Jupyter notebooks
├── scripts/             # Scripts directory
├── tests/               # Tests directory
├── outputs/             # Output directory
├── weight/              # Model weights directory
├── RedshiftPrediction.py # Redshift prediction script
├── train_aion_ddp_token.py # Training script
├── download_model.py    # Model download script
├── README.md            # Detailed documentation
├── README_CN.md         # Chinese detailed documentation
└── README_FIRST.md      # Quick start guide
```
