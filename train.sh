#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Add debug info
export NCCL_DEBUG=WARNING
export CUDA_LAUNCH_BLOCKING=1  # More synchronous CUDA operations
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth2
# Try different network interfaces or let NCCL auto-detect
# Fall back to TCP if needed
# Add more NCCL settings for stability
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=4
export TORCH_DISTRIBUTED_BACKEND=nccl
# Memory management
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_DISABLE=1
# Enable parallel CPU operations
export OMP_NUM_THREADS=8  # Parallel operations
export MKL_NUM_THREADS=8  # Intel MKL parallelism
export NUMEXPR_NUM_THREADS=8  # NumExpr parallelism

export NCCL_TIMEOUT=900            # allow up to 15 minutes
export NCCL_LAUNCH_TIMEOUT=600     # up to 10 minutes for the launch handshake



# Set initial batch size (will be divided by number of GPUs)
TOTAL_BATCH_SIZE=4  # Further reduce batch size
NUM_GPUS=4
PER_GPU_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NUM_GPUS))

# Directory paths
DATA_DIR="$HOME/prompt_image_segment/VQAv2"
# OUTPUT_DIR="$HOME/prompt_image_segment/outputs/debug_codebook_reduce_dim_512_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$HOME/prompt_image_segment/outputs/debug_nonanimal_animal_$(date +%Y%m%d_%H%M%S)"
RESUME_GEN_DIR="$HOME/prompt_image_segment/outputs/stage2_LLava_nonanimal_animal_20250427_231217/checkpoints/generator_epoch24_loss0.9583.pth"
RESUME_DIS_DIR="$HOME/prompt_image_segment/outputs/stage2_LLava_nonanimal_animal_20250427_231217/checkpoints/discriminator_epoch24_loss0.9583.pth"
RESUME_WEIGHT_DIR="$HOME/prompt_image_segment/outputs/debug_nonanimal_animal_20250505_003932/checkpoints/weight_module_epoch5_loss0.9602.pth"
# RESUME_GEN_DIR=None
# RESUME_DIS_DIR=None
STORE_DIR="$HOME/prompt_image_segment/stored_data/stage2_SNR16_embed_dim_256_None_None_20250415_192248"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Enable debug logging if needed
export LOGLEVEL=DEBUG

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Run training script with arguments
TRAIN_CMD="accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --num_machines 1 \
    main.py \
    --textalign\
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $PER_GPU_BATCH_SIZE \
    --num_epochs_1 40 \
    --num_epochs_2 40 \
    --num_epochs_3 40 \
    --start_epoch 0\
    --start_stage 2 \
    --learning_rate_g 1e-4 \
    --learning_rate_d 1e-5 \
    --learning_rate_w 1e-4 \
    --weight_decay 0.01 \
    --num_workers 1 \
    --discriminator_update_freq 1 \
    --lambda_sparsity 0.0 \
    --lambda_smoothness 0.0 \
    --lambda_answer 0.0 \
    --loss_recon 0.0 \
    --loss_perc 1.0 \
    --loss_vgg 0.2 \
    --loss_quant 0.5 \
    --loss_gen 0.5 \
    --loss_disc 0.5 \
    --SNR -15.0 \
    --log_interval 1000 \
    --sample_interval 100 \
    --train_category nonanimal \
    --val_category animal \
    --apply_weight 1 \
    --resume_generator_checkpoint $RESUME_GEN_DIR\
    --resume_discriminator_checkpoint $RESUME_DIS_DIR\
    --resume_weight_checkpoint $RESUME_WEIGHT_DIR\
    --generated_data_dir $STORE_DIR"

# Execute the training command
eval $TRAIN_CMD

# Save training command and arguments
echo "Training command and arguments:" > "$OUTPUT_DIR/training_args.txt"
echo "$TRAIN_CMD" >> "$OUTPUT_DIR/training_args.txt"