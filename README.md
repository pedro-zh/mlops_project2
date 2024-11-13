# MLops Project 2: Containerization

This project demonstrates how to containerize a machine learning application by creating a Docker image that runs a training script with optimized hyperparameters.
The project includes Python scripts for data processing, model training, and Docker configuration to ensure a reproducible environment both locally and on GitHub Codespaces.

## Project Overview

The goal of this project is to adapt the training notebook from the previous project into Python scripts, create a Docker image for running the training, and verify its functionality on GitHub Codespaces.

The project includes:

- Python scripts for data processing and model training (`data_module.py`, `model.py`, `main.py`)
- Docker configuration to build and run the training image
- A GitHub Codespaces configuration for seamless remote execution

## Repository Structure
```plaintext
mlops_project2/
│
├── data_module.py        # Contains the data pipeline (GLUEDataModule)
├── model.py              # Contains the model architecture (GLUETransformer)
├── main.py               # Main script to run the training with specified hyperparameters
├── dockerfile            # Dockerfile to build the training image
├── requirements.txt      # Python dependencies for the project
├── .devcontainer/        # GitHub Codespaces configuration folder
│   └── devcontainer.json
└── README.md             # Project documentation
```

## Prerequisites
- Docker installed on your local machine
- GitHub account
- Access to GitHub Codespaces (optional but recommended)

## Setup Instructions

### Clone the Repository
First, clone the repository to your local machine:
```sh
git clone https://github.com/pedro-zh/mlops_project2.git
cd mlops_project2
```

### Build the Docker Image
To build the Docker image locally, run the following command in the project directory:
```sh
docker build -t ml_training_image .
```
This will create a Docker image using the Dockerfile provided in this repository.

### Run the Docker Container
Once the image is built, you can run it with the following command:
```sh
docker run -it ml_training_image
```

### Experiment Tracking
The training logs and metrics are tracked using Weights & Biases (W&B). Ensure that you have access to the tool to monitor the training progress and results. If you do not have access to W&B, you can proceed with the prompts in the terminal without tracking.

### Running in GitHub Codespaces

1. Open this repository in GitHub Codespaces by clicking on the "Code" button and selecting "Open with Codespaces."
2. The `.devcontainer/devcontainer.json` configuration will automatically set up the environment for you.
3. Once the Codespaces environment is set up, you can build the Docker image using the same command as above and run the training.
