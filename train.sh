#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 4 GPUs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Add debug info
export NCCL_DEBUG=WARNING
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues
export CUDA_LAUNCH_BLOCKING=1  # More synchronous CUDA operations
# Try different network interfaces or let NCCL auto-detect
# Fall back to TCP if needed
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0  # Disable RDMA
# Add more NCCL settings for stability
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=4
# Memory management
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_DISABLE=1
# Enable parallel CPU operations
export OMP_NUM_THREADS=8  # Parallel operations
export MKL_NUM_THREADS=8  # Intel MKL parallelism
export NUMEXPR_NUM_THREADS=8  # NumExpr parallelism


# Set initial batch size (will be divided by number of GPUs)
TOTAL_BATCH_SIZE=9  # Further reduce batch size
NUM_GPUS=3
PER_GPU_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NUM_GPUS))

# Directory paths
DATA_DIR="$HOME/prompt_image_segment/VQAv2"
OUTPUT_DIR="$HOME/prompt_image_segment/outputs/debug_codebook_reduce_dim_512_$(date +%Y%m%d_%H%M%S)"
# OUTPUT_DIR="$HOME/prompt_image_segment/outputs/20250225_164551"
# RESUME_DIR="$HOME/prompt_image_segment/outputs/codebook_reduce_dim_512_20250304_231036/checkpoints/checkpoint_epoch_54_loss_1.5045.pth"
RESUME_DIR=None

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Enable debug logging if needed
export LOGLEVEL=DEBUG

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Run training script with arguments
TRAIN_CMD="accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 1 \
    --num_machines 1 \
    main.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $PER_GPU_BATCH_SIZE \
    --num_epochs 400 \
    --start_epoch 0 \
    --learning_rate_g 1e-4 \
    --learning_rate_d 1e-5 \
    --weight_decay 0.01 \
    --num_workers 1 \
    --discriminator_update_freq 2 \
    --lambda_sparsity 0.0 \
    --lambda_smoothness 0.0 \
    --lambda_answer 0.0 \
    --loss_recon 1.0 \
    --loss_perc 0.0 \
    --loss_vgg 10.0 \
    --loss_quant 1.0 \
    --loss_gen 0.0 \
    --loss_disc 0.5 \
    --log_interval 200 \
    --sample_interval 100 \
    --split_type image_based \
    --split_level2 subject \
    --train_category animal \
    --val_category human \
    --resume_from_checkpoint $RESUME_DIR"

# Execute the training command
eval $TRAIN_CMD

# Save training command and arguments
echo "Training command and arguments:" > "$OUTPUT_DIR/training_args.txt"
echo "$TRAIN_CMD" >> "$OUTPUT_DIR/training_args.txt"