import torch
from PIL import Image
import requests
from transformers import AutoProcessor
from model import VQAWithSegmentation, PatchGANDiscriminator
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import wandb
import json
from datetime import datetime, timedelta
from dataset import VQAv2Dataset, create_vqa_dataloaders
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
import torch.cuda
import gc
from accelerate import DistributedDataParallelKwargs
import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb  
from torch.cuda.amp import autocast, GradScaler
import argparse
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from model_config import ModelConfig
import netifaces  # Add this import at the top
import threading
import time  # Add at top of file
import sys
import logging
from args import parse_args
from utils import (
    setup_logger, 
    print_network_info, 
    print_memory_stats, 
    save_checkpoint, 
    load_checkpoint,
    setup_distributed_training,
    cleanup_distributed,
    visualize_batch
)
import math


def calculate_segmentation_loss(pred_masks, target_masks):
    """Calculate segmentation loss using binary cross entropy"""
    # Ensure same shape
    if pred_masks.shape != target_masks.shape:
        target_masks = torch.nn.functional.interpolate(
            target_masks.unsqueeze(1).float(), 
            size=pred_masks.shape[-2:],
            mode='nearest'
        ).squeeze(1)
    
    # Binary cross entropy loss
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_masks, 
        target_masks.float(),
        reduction='mean'
    )
    
    return bce_loss

def calculate_sparsity_loss(masks):
    """Encourage sparse segmentation masks"""
    # L1 regularization to encourage sparsity
    return torch.mean(torch.abs(masks))

def calculate_smoothness_loss(masks):
    """Encourage smooth segmentation masks"""
    # Calculate gradients
    dy = masks[:, :, 1:] - masks[:, :, :-1]
    dx = masks[:, :, :, 1:] - masks[:, :, :, :-1]
    
    # Calculate total variation loss
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

def calculate_answer_relevance_loss(original_outputs, masked_outputs):
    """Calculate loss based on answer relevance"""
    # Calculate MSE loss
    criterion = nn.MSELoss()  
    
    # Calculate loss between masked outputs and original outputs
    loss = criterion(original_outputs, masked_outputs)
    
    return loss

def is_answer_correct(pred_answer, gt_answer):
    """Check if predicted answer matches ground truth"""
    # Simple exact match for now
    # Could be extended with more sophisticated metrics
    return pred_answer.lower().strip() == gt_answer.lower().strip()

def calculate_total_loss(outputs, args):
    """Calculate total loss combining segmentation, sparsity, smoothness and answer losses"""
    # Segmentation loss
    # segmentation_loss = calculate_segmentation_loss(
    #     outputs['segmentation_masks'], 
    #     batch['target_mask'].to(outputs['segmentation_masks'].device)
    # )
    answer_loss = calculate_answer_relevance_loss(
        outputs['logits'],
        outputs['masked_logits'],
    )
    
    # Sparsity loss to encourage sparse masks
    sparsity_loss = calculate_sparsity_loss(outputs['segmentation_masks'])
    
    # Smoothness loss to encourage smooth masks
    # smoothness_loss = calculate_smoothness_loss(outputs['segmentation_masks'])
    
    # Combine losses with weights
    total_loss = (
        args.lambda_answer * answer_loss + 
        args.lambda_sparsity * sparsity_loss
    )
    
    return total_loss

def feature_matching_loss(discriminator, real_images, fake_images, loss_fn=F.l1_loss):
    """
    Computes the feature matching loss between real and fake images using the discriminator's intermediate features.
    Assumes discriminator.get_intermediate_features(x) returns a list of intermediate feature maps.
    """
    # Get real features (detached so gradients do not flow into D for real images)
    with torch.no_grad():
        real_feats = discriminator.get_intermediate_features(real_images)
        real_feats = [f.detach() for f in real_feats]
    # Get fake features (allow gradients to flow into generator)
    fake_feats = discriminator.get_intermediate_features(fake_images)
    
    fm_loss = 0.0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        fm_loss += loss_fn(fake_feat, real_feat)
    return fm_loss

def train_epoch(generator, discriminator, train_dataloader, optimizer, epoch, device, args, accelerator):
    """Train for one epoch with separate generator and discriminator steps"""
    torch.autograd.set_detect_anomaly(True)
    
    logger = logging.getLogger('training')
    generator.train()
    total_g_loss = 0
    total_d_loss = 0
    total_a_loss = 0
    total_disc_update = 0
    
    optimizer_G, optimizer_D = optimizer  # Unpack optimizers

    # Define adversarial loss
    adv_loss = nn.BCEWithLogitsLoss()

    # Create progress bar
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Training Epoch {epoch}",
            position=0,
            leave=True,
            file=sys.stdout
        )
   
    try:
        for batch_idx, batch in enumerate(train_dataloader):
            # Train Discriminator only on even batch indices
            optimizer_G.zero_grad()
            with autocast():
                outputs = generator(batch['image'], batch['question'])
                generated_images = outputs['generated_images']
                generated_images = torch.clamp(generated_images, 0, 1)
                fake_pred = discriminator(generated_images)
                g_loss_adv = adv_loss(fake_pred, torch.ones_like(fake_pred))

                fm_loss = feature_matching_loss(discriminator, batch['image'], generated_images)
                
                
                g_loss = (
                    args.loss_recon * outputs['loss_recon'] +  
                    args.loss_vgg * outputs['loss_vgg'] +
                    args.loss_gen * (g_loss_adv + fm_loss)
                )
            
            accelerator.backward(g_loss)
            optimizer_G.step()
            with autocast():
                # Discriminator predictions
                real_pred = discriminator(batch['image'])
                fake_pred = discriminator(generated_images.detach())
                
                # Calculate discriminator losses
                d_loss_real = adv_loss(real_pred, torch.ones_like(real_pred))
                d_loss_fake = adv_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (d_loss_real + d_loss_fake) * 0.5

            should_update_dis = g_loss_adv.item() <= 0.65 or d_loss.item() >= 0.65
            should_update_dis = accelerator.gather_for_metrics(torch.tensor(should_update_dis, device=device))
            should_update_dis = should_update_dis.any().item()  # Ensure all ranks agree

            if should_update_dis:
                total_disc_update += 1
                optimizer_D.zero_grad()
                
                accelerator.backward(d_loss)
                optimizer_D.step()

            total_g_loss += g_loss.item()  # Track generator loss
            total_a_loss += g_loss_adv.item()  # Track VGG loss
            total_d_loss += d_loss.item()  # Track discriminator loss
            
            # Update progress bar on main process
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "A_loss": f"{g_loss_adv.item():.4f}",
                    "FM_loss": f"{fm_loss.item():.4f}",
                    "D_loss": f"{d_loss.item():.4f}"
                })
                progress_bar.update(1)
            
            # Log batch statistics only on main process
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                with torch.no_grad():
                    visualize_batch(
                        batch['image'][0],
                        outputs['generated_images'],
                        batch['question'][0],
                        batch['answer_text'][0],
                        epoch,
                        batch_idx,
                        args,
                        mode='train'
                    )
                    torch.cuda.empty_cache()
            
            
            # Synchronize after batch processing
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                logger.debug(
                    "Batch %d stats - Loss components: recon=%.4f, perc=%.4f, vgg=%.4f, g_loss=%.4f, fm_loss=%4f, d_loss=%.4f",
                    batch_idx,
                    outputs["loss_recon"],
                    outputs["loss_perc"],
                    outputs["loss_vgg"],
                    g_loss_adv.item(),
                    fm_loss.item(),
                    d_loss.item()
                )  
        if accelerator.is_main_process :
            logger.debug(
                "Epoch %d stats - Disc updates: %d",
                epoch,
                total_disc_update
            ) 
    except Exception as e:
        print(f"Error in train_epoch: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
    
    # Set gradient clipping
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    
    return total_g_loss / len(train_dataloader), total_a_loss / len(train_dataloader), total_d_loss / len(train_dataloader)

def validate(generator, val_loader, epoch, device, args, accelerator):
    """Validate the model"""
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            with autocast():
                outputs = generator(batch['image'], batch['question'])
                loss = (
                    args.loss_recon * outputs['loss_recon'] + 
                    args.loss_perc * outputs['loss_perc'] + 
                    args.loss_vgg * outputs['loss_vgg']
                )
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def main(args):
    # Setup logger
    logger = setup_logger(args)
    logger.info("Starting training with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Print network information and get valid interface
    valid_interface = None
    if int(os.environ.get("LOCAL_RANK", -1)) == 0:
        valid_interface = print_network_info()
    
    # Share the interface name across processes
    if torch.distributed.is_initialized():
        if int(os.environ.get("LOCAL_RANK", -1)) == 0:
            interface_tensor = torch.tensor([ord(c) for c in valid_interface], dtype=torch.long).cuda()
        else:
            interface_tensor = torch.zeros(20, dtype=torch.long).cuda()  # Assume max 20 chars
        torch.distributed.broadcast(interface_tensor, 0)
        valid_interface = ''.join([chr(i) for i in interface_tensor.tolist() if i != 0])
    
    # Set NCCL socket interface
    if valid_interface:
        os.environ["NCCL_SOCKET_IFNAME"] = valid_interface
        print(f"Using network interface: {valid_interface}")
    
    # Load configuration from config.json
    config = ModelConfig.from_json("config.json")
    print("Loaded Configuration:")
    
    # Get local rank for device placement
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(f'cuda:{local_rank}' if local_rank != -1 else 'cuda')
    
    # Initialize process group with TCP backend as fallback
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        # Try NCCL first with explicit socket interface
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=5)
        )
    except Exception as e:
        print(f"NCCL initialization failed: {e}")
        print("Falling back to TCP backend...")
        # Fall back to TCP backend
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(minutes=5)
        )
    
    # Configure FSDP Plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        sharding_strategy="NO_SHARD",    # Disable sharding for now
        cpu_offload=False,
        use_orig_params=True,            # Use original parameters
    )

    # Initialize accelerator with FSDP
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        device_placement=True,
    )
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="prompt_image_segment",
            dir=args.output_dir,
            config=args,
            name=run_name,
            resume="allow"
        )
    
    # Make sure wandb is initialized before proceeding
    accelerator.wait_for_everyone()
    
    # Initialize models
    generator = VQAWithSegmentation(
        config=config,
        device=device,
    )
    discriminator = PatchGANDiscriminator(
        in_channels=3, 
        base_channels=64
    ).to(device)
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Create separate optimizers
    optimizer_G = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=args.learning_rate_g,
        weight_decay=args.weight_decay
    )
    
    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.learning_rate_d,
        weight_decay=args.weight_decay
    )

    # Create cosine annealing scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Keep initial learning rate during warmup
            return 1.0
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)
            return 0.01 + (1 - 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    # Load checkpoint if it exists
    start_epoch = args.start_epoch
    best_loss = float('inf')
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        load_checkpoint(generator, optimizer_G, args.resume_from_checkpoint, device)
        
        # Override loaded start_epoch if specified in args
        if args.start_epoch > 0:
            start_epoch = args.start_epoch

    
    # Prepare dataloaders
    train_dataloader, val_dataloader = create_vqa_dataloaders(args)
    
    # Let accelerator handle distribution
    generator, discriminator, optimizer_G, optimizer_D, train_dataloader, val_dataloader = accelerator.prepare(
        generator, discriminator, optimizer_G, optimizer_D, train_dataloader, val_dataloader
    )
    
    
    try:
        # Training loop
        for epoch in range(start_epoch, args.num_epochs):
        
            with autocast():
                generator.train()
                train_g_loss, train_a_loss, train_d_loss = train_epoch(generator, discriminator, train_dataloader, [optimizer_G, optimizer_D], epoch, device, args, accelerator)
            
            
            # Run validation every 5 epochs
            generator.eval()
            val_loss = validate(generator, val_dataloader, epoch, device, args, accelerator)
            logger.info(f"Epoch {epoch} - Train G loss: {train_g_loss:.4f}, Train A loss: {train_a_loss:.4f}, Train D loss: {train_d_loss:.4f}, Val loss: {val_loss:.4f}")
            
            scheduler_G.step()
            scheduler_D.step()
            
            # Log learning rate
            if accelerator.is_main_process:
                current_lr_G = optimizer_G.param_groups[0]['lr']
                current_lr_D = optimizer_D.param_groups[0]['lr']
                wandb.log({
                    'epoch': epoch,
                    'learning_rate_G': current_lr_G,
                    'learning_rate_D': current_lr_D,
                    'epoch_train_g_loss': train_g_loss,
                    'epoch_train_a_loss': train_a_loss,
                    'epoch_train_d_loss': train_d_loss,
                    'epoch_val_loss': val_loss,
                })
                
                # Save checkpoint asynchronously
                logger.info(f"Saving checkpoint with val_loss: {val_loss}")
                save_thread = threading.Thread(
                    target=save_checkpoint,
                    args=(
                        generator.module if hasattr(generator, "module") else generator,
                        epoch,
                        val_loss,  # Make sure this is not None
                        args
                    )
                )
                save_thread.start()
                
                # Wait for saving to complete before synchronization
                save_thread.join()
                logger.info(f"Rank 0: Checkpoint saving completed")

            
            torch.cuda.empty_cache()  # Clear memory before synchronization
            
            # All processes wait here once per epoch with longer timeout
            accelerator.wait_for_everyone() 
            
            
    except Exception as e:
        print(f"Training error: {str(e)}")
        if torch.distributed.is_initialized():
            try:
                accelerator.wait_for_everyone()
            except:
                pass
            torch.distributed.destroy_process_group()
        raise e
    finally:
        if accelerator.is_main_process and wandb.run is not None:
            wandb.finish()
        cleanup_distributed()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)