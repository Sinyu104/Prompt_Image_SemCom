# Prompt Image Segmentation with VQA

This project implements a Visual Question Answering (VQA) system with image segmentation capabilities. The model can answer questions about images while highlighting relevant regions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/prompt_image_segment.git
cd prompt_image_segment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the VQAv2 dataset:
```bash
# Create data directory
mkdir -p VQAv2
cd VQAv2

# Download dataset files
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# Download COCO Images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# Unzip all files
unzip "*.zip"

# Clean up zip files
rm *.zip

# Expected directory structure:
# VQAv2/
# ├── train2014/                     # Training images
# ├── val2014/                       # Validation images
# ├── v2_OpenEnded_mscoco_train2014_questions.json
# ├── v2_OpenEnded_mscoco_val2014_questions.json
# ├── v2_mscoco_train2014_annotations.json
# └── v2_mscoco_val2014_annotations.json
```

## Usage

### Training

To train the model, use the `train.sh` script:

```bash
./train.sh
```

Or run with specific arguments:

```bash
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision fp16 \
    main.py \
    --data_dir /path/to/VQAv2 \
    --output_dir outputs \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4
```

### Command Line Arguments

The following arguments can be specified:

#### Training Parameters
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation (default: 16)
- `--mixed_precision`: Mixed precision training mode ["no", "fp16", "bf16"] (default: "fp16")
- `--learning_rate`: Initial learning rate (default: 5e-5)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 0.01)
- `--num_epochs`: Number of training epochs (default: 10)
- `--start_epoch`: Starting epoch number for resuming training (default: 0)

#### Dataset and Model Arguments
- `--batch_size`: Batch size per GPU (default: 32)
- `--output_dir`: Directory to save outputs (default: "outputs")
- `--data_dir`: Path to VQA v2 dataset (required)
- `--num_workers`: Number of data loading workers (default: 4)
- `--resume_from_checkpoint`: Path to checkpoint for resuming training (default: None)

#### Loss Weights
- `--lambda_sparsity`: Weight for sparsity loss (default: 0.1)
- `--lambda_smoothness`: Weight for smoothness loss (default: 0.05)
- `--lambda_answer`: Weight for answer relevance loss (default: 1.0)
- `--loss_recon`: Weight for reconstruction loss (default: 1.0)
- `--loss_perc`: Weight for perceptual loss (default: 0.1)

#### Logging Arguments
- `--log_interval`: How often to log training stats (default: 10)
- `--sample_interval`: How often to save example predictions (default: 100)
- `--vis_interval`: How often to save visualizations (default: 100)

### Resuming Training

To resume training from a checkpoint:

```bash
accelerate launch main.py \
    --resume_from_checkpoint path/to/checkpoint.pth \
    --start_epoch 5 \
    --data_dir /path/to/VQAv2
```

## Monitoring

Training progress can be monitored through:

### Output Directory Structure
```
outputs/
├── YYYYMMDD_HHMMSS/              # Timestamp-based run directory
│   ├── training_YYYYMMDD_HHMMSS.log  # Training logs
│   ├── training_args.txt         # Command line arguments used
│   ├── checkpoints/              # Model checkpoints
│   │   ├── checkpoint_epoch_0_loss_X.XXXX.pth
│   │   ├── checkpoint_epoch_1_loss_X.XXXX.pth
│   │   └── ...
│   ├── visualizations/           # Generated images and visualizations
│   │   ├── train/               # Training visualizations
│   │   │   ├── epoch_0/
│   │   │   │   ├── batch_0.png
│   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── val/                 # Validation visualizations
│   │       ├── epoch_0/
│   │       │   ├── batch_0.png
│   │       │   └── ...
│   │       └── ...
│   └── wandb/                   # Weights & Biases logging
└── ...
```

### Monitoring Tools

#### 1. Training Logs
- `training_YYYYMMDD_HHMMSS.log`: Contains detailed training information including:
  - Loss values for each batch
  - Learning rates
  - GPU memory usage
  - Training/validation metrics

#### 2. Checkpoints
- Saved every epoch with format `checkpoint_epoch_{epoch}_loss_{loss}.pth`
- Contains:
  - Model state dict
  - Optimizer state
  - Current epoch
  - Best loss value

#### 3. Visualizations
- Generated every `sample_interval` batches
- Organized by:
  - Training/validation split
  - Epoch number
  - Shows:
    - Original image
    - Generated image
    - Question and ground truth answer

#### 4. Weights & Biases Integration
- Real-time monitoring of:
  - Training/validation losses
  - Learning rate changes
  - GPU utilization
  - Generated image samples
  - Custom metrics
- Access through: `https://wandb.ai/your-username/prompt_image_segment`

## License

MIT License


## Citation

If you use this code in your research, please cite:

```bibtex
[Your citation information]
``` 