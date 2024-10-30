# Use the official PyTorch image with CUDA 12.4 and cuDNN 9
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the training script
CMD ["python", "main.py", "--checkpoint_dir", "models", "--lr", "2e-05", "--weight_decay", "0.005", "--warmup_steps", "250"]
