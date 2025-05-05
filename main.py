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
import torch.autograd
from args import parse_args
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from JPG import JPGTransmission
from utils import (
    setup_logger, 
    print_network_info, 
    print_memory_stats, 
    save_checkpoint, 
    load_checkpoint,
    setup_distributed_training,
    cleanup_distributed,
    visualize_batch,
    store_generated_outputs,
    save_reconstructed_image
)
import math
import socket, psutil
from itertools import islice

def pick_real_interface():
    """
    Return the first non-loopback, non-link-local IPv4 interface name,
    or None if none found.
    """
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip = addr.address
                # skip loopback and link-local
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                # we’ve found a usable interface!
                print(f"Using interface {iface} with IP {ip}")
                return iface
    print("No non-loopback interface found")
    return None

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

# def unwrap_fsdp(module):
#     return module.module if isinstance(module, FSDP) else module

# def gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10.0):
#     # Unwrap FSDP to access the actual model
#     raw_discriminator = unwrap_fsdp(discriminator)

#     # Save current training mode
#     was_training = raw_discriminator.training
#     raw_discriminator.eval()

    
#     batch_size = real_data.size(0)
#     real_data = real_data.detach()
#     fake_data = fake_data.detach()
#     alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_data)
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#     if not torch.isfinite(interpolates).all():
#         raise ValueError("Interpolates contain NaNs or Infs.")
    
#     with torch.autograd.set_detect_anomaly(True):
#         interpolates = interpolates.to(device).requires_grad_(True)
#         interpolates = torch.clamp(interpolates, 0, 1)
#         d_interpolates = raw_discriminator(interpolates)
#         d_interpolates = d_interpolates.view(batch_size, -1)  # [batch_size, num_patches]

#     try:
#         fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
        
#         gradients = torch.autograd.grad(
#             outputs=d_interpolates,
#             inputs=interpolates,
#             grad_outputs=fake,  # Now grad_outputs is defined
#             create_graph=False,
#             retain_graph=True,
#             only_inputs=True,
#         )[0]
        
#         # Reshape and compute gradient norm
#         gradients = gradients.reshape(batch_size, -1)
#         grad_norm = gradients.norm(2, dim=1)
#         # Compute gradient penalty
#         gradient_penalty = ((grad_norm - 1) ** 2).mean()
#         # Restore original mode
#         if was_training:
#             raw_discriminator.train()
#         return gradient_penalty
    
#     except Exception as e:
#         print(f"Error computing gradient penalty: {str(e)}")
#         # Return a differentiable zero tensor
#         return torch.tensor(0.0, device=device, requires_grad=True)

def gradient_penalty(critic, real_data, fake_data, device, lambda_gp=10.0, logger=None):
    
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    d_interpolates = d_interpolates.view(batch_size, -1)  # [batch_size, num_patches]

    try:
        fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,  # Now grad_outputs is defined
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Reshape and compute gradient norm
        gradients = gradients.reshape(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        if logger:
            logger.debug(f"[GP] grad_norm stats: μ={grad_norm.mean().item():.4f}, σ={grad_norm.std().item():.4f}, "
                f"min={grad_norm.min().item():.4f}, max={grad_norm.max().item():.4f}")

        # Compute gradient penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()/gradients.shape[1]
        
        # Ensure penalty requires grad
        if not gradient_penalty.requires_grad:
            gradient_penalty.requires_grad_(True)
            
        return gradient_penalty
        
    except Exception as e:
        print(f"Error computing gradient penalty: {str(e)}")
        # Return a differentiable zero tensor
        return torch.tensor(0.0, device=device, requires_grad=True)
   
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

def stage1_train(generator, discriminator, train_dataloader, optimizer, epoch, device, args, accelerator, phase=1):
    """Train for one epoch with separate generator and discriminator steps"""
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    
    logger = logging.getLogger('training')
    generator.train()
    

    optimizer_G, optimizer_D = optimizer  # Unpack optimizers

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
        if phase == 1:
            total_g_loss = 0
            total_q_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Train Discriminator only on even batch indices
                optimizer_G.zero_grad()
                with autocast(enabled=False):
                    outputs = generator(batch['image'], batch['question'], batch['answer_text'], stage=1, textalign=args.textalign)
                    generated_images = outputs['generated_images']
                    generated_images = torch.clamp(generated_images, 0, 1)
        
                    g_loss = (
                        args.loss_recon * outputs['loss_recon'] +
                        args.loss_vgg * outputs['loss_vgg'] +
                        args.loss_perc   * outputs['loss_perc'] +
                        args.loss_quant * outputs['quantization_loss'] 
                    )
                    logger.debug(f"loss_recon: {outputs['loss_recon'].item()}, loss_perc : {outputs['loss_perc'].item()}, loss_vgg: {outputs['loss_vgg'].item()}, loss_quant: {outputs['quantization_loss']} ")
                    

                accelerator.backward(g_loss)
                
                # Clip gradients for generator
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()


                total_g_loss += g_loss.item()  # Track generator loss
                total_q_loss += outputs['quantization_loss'].item()  # Track quantization loss
                
                # Update progress bar on main process
                if accelerator.is_main_process:
                    progress_bar.set_postfix({
                        "G_loss": f"{g_loss.item():.4f}"
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
                        "Batch %d stats - Loss components: recon=%.4f, perc=%.4f, vgg=%.4f",
                        batch_idx,
                        outputs["loss_recon"],
                        outputs["loss_perc"],
                        outputs["loss_vgg"],
                    )  

            return total_g_loss / len(train_dataloader), total_q_loss / len(train_dataloader)

        elif phase ==2:
            total_g_loss = 0
            total_d_loss = 0
            total_a_loss = 0
            total_q_loss = 0
            total_p_loss = 0
            lambda_gp = 1.0
            for batch_idx, batch in enumerate(train_dataloader):
                # Train Discriminator only on even batch indices
                optimizer_G.zero_grad()
                with autocast(enabled=False):
                    outputs = generator(batch['image'], batch['question'], batch['answer_text'], stage=1, textalign=args.textalign)
                    generated_images = outputs['generated_images']
                    generated_images = torch.clamp(generated_images, 0, 1)
                    # Compute the critic (discriminator) score on fake images
                    fake_score = discriminator(generated_images)
                    # WGAN generator loss: maximize fake_score, so minimize negative score
                    g_loss_adv = -fake_score.mean()
                    logger.debug(f"g_loss_adv : {g_loss_adv.item()}, loss_perc : {outputs['loss_perc'].item()}, loss_vgg : {outputs['loss_vgg'].item()}, loss_quant : {outputs['quantization_loss'] .item()}")
                    
                    # Total generator loss: include reconstruction, VGG, adversarial (WGAN) and feature matching losses
                    g_loss = (
                        args.loss_recon * outputs['loss_recon'] +
                        args.loss_perc   * outputs['loss_perc'] +
                        args.loss_vgg   * outputs['loss_vgg'] +
                        args.loss_gen   * g_loss_adv +
                        args.loss_quant * outputs['quantization_loss'] 
                    )
                accelerator.backward(g_loss)
                
                # Clip gradients for generator
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

                if batch_idx % args.discriminator_update_freq  == 0:
                    # Train Discriminator
                    optimizer_D.zero_grad()
                    with autocast(enabled=False):
                        # Compute scores on real and fake images
                        real_score = discriminator(batch['image'])
                        fake_score = discriminator(generated_images.detach())
                        # WGAN discriminator loss: maximize (real_score - fake_score), so minimize negative of that
                        d_loss_adv = -(real_score.mean() - fake_score.mean())

                        logger.debug(f"real_score : {real_score.mean().item()}, fake_score: {fake_score.mean().item()}")
                        # Compute gradient penalty
                        gp = gradient_penalty(discriminator, batch['image'], generated_images.detach(), device, lambda_gp=lambda_gp, logger=logger)
                        
                        logger.debug(f"Gradient penalty: {gp}, Discriminator loss: {d_loss_adv}")
                        
                        # Total discriminator loss: adversarial loss + weighted gradient penalty
                        d_loss = d_loss_adv+ lambda_gp * gp
                        accelerator.backward(d_loss)
                        # Clip gradients for discriminator
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizer_D.step()

                total_g_loss += g_loss.item()   # Track generator loss
                total_a_loss += g_loss_adv.item()   # Track VGG loss
                total_d_loss += d_loss.item()  # Track discriminator loss
                total_p_loss += outputs['loss_perc'].item()
                total_q_loss += outputs['quantization_loss'].item()   # Track quantization loss
                
                # Update progress bar on main process
                if accelerator.is_main_process:
                    progress_bar.set_postfix({
                        "G_loss": f"{g_loss.item():.4f}",
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
                
                if accelerator.is_main_process and batch_idx % args.log_interval :
                    logger.debug(
                        "Batch %d stats - Loss components: recon=%.4f, perc=%.4f, vgg=%.4f, adv=%.4f, d=%.4f, gp=%.4f",
                        batch_idx,
                        outputs["loss_recon"].item(),
                        outputs["loss_perc"],
                        outputs["loss_vgg"].item(),
                        g_loss_adv.item(),
                        d_loss.item(),
                        gp.item(),
                    )  

        else:
            Exception("Invalid stage for training")
        
        
    except Exception as e:
        print(f"Error in train_epoch: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
    
    # Set gradient clipping
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    
    return total_g_loss / len(train_dataloader), total_a_loss / len(train_dataloader), total_d_loss / len(train_dataloader), total_q_loss / len(train_dataloader), total_p_loss / len(train_dataloader)

def stage2_train(generator, discriminator, train_dataloader, optimizer, epoch, device, args, accelerator):
    """
    Stage 2 Training: Optimize RL-weighting agent (fix encoder/decoder).
    
    Args:
        generator: the full model (encoder + decoder + rl_agent)
        discriminator: not used in Stage 2 (optional)
        train_dataloader: training data loader
        optimizers: [optimizer_RL]
        epoch: current epoch number
        device: device
        args: training args
        accelerator: HuggingFace accelerator
    Returns:
        train_rl_loss: total loss for the RL agent (typically derived from task loss)
    """
    logger = logging.getLogger('training')
    generator.train()
    

    optimizer_G, optimizer_D = optimizer  # Unpack optimizers

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

        total_g_loss = 0
        total_d_loss = 0
        total_a_loss = 0
        total_q_loss = 0
        total_p_loss = 0
        lambda_gp = 1.0
        for batch_idx, batch in enumerate(train_dataloader):
            # Train Discriminator only on even batch indices
            optimizer_G.zero_grad()
            with autocast(enabled=False):
                outputs = generator(batch['image'], batch['question'], batch['answer_text'], stage=2, textalign=args.textalign)
                generated_images = outputs['generated_images']
                generated_images = torch.clamp(generated_images, 0, 1)
                # Compute the critic (discriminator) score on fake images
                fake_score = discriminator(generated_images)
                # WGAN generator loss: maximize fake_score, so minimize negative score
                g_loss_adv = -fake_score.mean()
                logger.debug(f"g_loss_adv : {g_loss_adv.item()}, loss_perc : {outputs['loss_perc'].item()}, loss_vgg : {outputs['loss_vgg'].item()}, loss_quant : {outputs['quantization_loss'] .item()}")
                
                # Total generator loss: include reconstruction, VGG, adversarial (WGAN) and feature matching losses
                g_loss = (
                    args.loss_recon * outputs['loss_recon'] +
                    args.loss_perc   * outputs['loss_perc'] +
                    args.loss_vgg   * outputs['loss_vgg'] +
                    args.loss_gen   * g_loss_adv +
                    args.loss_quant * outputs['quantization_loss'] 
                )
            accelerator.backward(g_loss)
            
            # Clip gradients for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            if batch_idx % args.discriminator_update_freq  == 0:
                # Train Discriminator
                optimizer_D.zero_grad()
                with autocast(enabled=False):
                    # Compute scores on real and fake images
                    real_score = discriminator(batch['image'])
                    fake_score = discriminator(generated_images.detach())
                    # WGAN discriminator loss: maximize (real_score - fake_score), so minimize negative of that
                    d_loss_adv = -(real_score.mean() - fake_score.mean())

                    logger.debug(f"real_score : {real_score.mean().item()}, fake_score: {fake_score.mean().item()}")
                    # Compute gradient penalty
                    gp = gradient_penalty(discriminator, batch['image'], generated_images.detach(), device, lambda_gp=lambda_gp, logger=logger)
                    
                    logger.debug(f"Gradient penalty: {gp}, Discriminator loss: {d_loss_adv}")
                    
                    # Total discriminator loss: adversarial loss + weighted gradient penalty
                    d_loss = d_loss_adv+ lambda_gp * gp
                    accelerator.backward(d_loss)
                    # Clip gradients for discriminator
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_D.step()

            total_g_loss += g_loss.item()   # Track generator loss
            total_a_loss += g_loss_adv.item()   # Track VGG loss
            total_d_loss += d_loss.item()  # Track discriminator loss
            total_p_loss += outputs['loss_perc'].item() # Track LLaVA loss
            total_q_loss += outputs['quantization_loss'].item()   # Track quantization loss
            
            # Update progress bar on main process
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "G_loss": f"{g_loss.item():.4f}",
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
            
            if accelerator.is_main_process and batch_idx % args.log_interval :
                logger.debug(
                    "Batch %d stats - Loss components: recon=%.4f, perc=%.4f, vgg=%.4f, adv=%.4f, d=%.4f, gp=%.4f",
                    batch_idx,
                    outputs["loss_recon"].item(),
                    outputs["loss_perc"],
                    outputs["loss_vgg"].item(),
                    g_loss_adv.item(),
                    d_loss.item(),
                    gp.item(),
                )  

    except Exception as e:
        print(f"Error in train_epoch: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
    
    # Set gradient clipping
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    
    return total_g_loss / len(train_dataloader), total_a_loss / len(train_dataloader), total_d_loss / len(train_dataloader), total_q_loss / len(train_dataloader), total_p_loss / len(train_dataloader)


def train_weight_module(generator, discriminator, train_dataloader, optimizer, epoch, device, args, accelerator):
    """
    Stage 2 Training: Optimize RL-weighting agent (fix encoder/decoder).
    
    Args:
        generator: the full model (encoder + decoder + rl_agent)
        discriminator: not used in Stage 2 (optional)
        train_dataloader: training data loader
        optimizers: [optimizer_RL]
        epoch: current epoch number
        device: device
        args: training args
        accelerator: HuggingFace accelerator
    Returns:
        train_rl_loss: total loss for the RL agent (typically derived from task loss)
    """
    logger = logging.getLogger('training')
    generator.train()
    

    optimizer_W = optimizer  # Unpack optimizers

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

        total_policy_loss = 0.0
        total_regret = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            # Train Discriminator only on even batch indices
            if batch_idx > 0:
                break
            optimizer_W.zero_grad()
            with autocast(enabled=False):
                outputs = generator.forward_with_regret(batch['image'], batch['question'], batch['answer_text'], stage=2, textalign=args.textalign)
                generated_images_sample = outputs['generated_images_sample']
                generated_images_sample = torch.clamp(generated_images_sample, 0, 1)
                # Compute the critic (discriminator) score on fake images
                fake_score_sample = discriminator(generated_images_sample)
                # WGAN generator loss: maximize fake_score, so minimize negative score
                g_loss_adv_sample = -fake_score_sample.mean()
                logger.debug(f"g_loss_adv_sample : {g_loss_adv_sample.item()}, loss_perc_sample : {outputs['loss_perc_sample'].item()}, loss_vgg_sample : {outputs['loss_vgg_sample'].item()}, loss_quant_sample : {outputs['quantization_loss_sample'] .item()}")
                
                # Total generator loss: include reconstruction, VGG, adversarial (WGAN) and feature matching losses
                g_loss_sample = (
                    args.loss_recon * outputs['loss_recon_sample'] +
                    args.loss_perc   * outputs['loss_perc_sample'] +
                    args.loss_vgg   * outputs['loss_vgg_sample'] +
                    args.loss_gen   * g_loss_adv_sample +
                    args.loss_quant * outputs['quantization_loss_sample'] 
                )
                
                generated_images_det = outputs['generated_images_det']
                generated_images_det = torch.clamp(generated_images_det, 0, 1)
                # Compute the critic (discriminator) score on fake images
                fake_score_det = discriminator(generated_images_det)
                # WGAN generator loss: maximize fake_score, so minimize negative score
                g_loss_adv_det = -fake_score_det.mean()
                logger.debug(f"g_loss_adv_det : {g_loss_adv_det.item()}, loss_perc_det : {outputs['loss_perc_det'].item()}, loss_vgg_det : {outputs['loss_vgg_det'].item()}, loss_quant_det : {outputs['quantization_loss_det'] .item()}")
                
                # Total generator loss: include reconstruction, VGG, adversarial (WGAN) and feature matching losses
                g_loss_det = (
                    args.loss_recon * outputs['loss_recon_det'] +
                    args.loss_perc   * outputs['loss_perc_det'] +
                    args.loss_vgg   * outputs['loss_vgg_det'] +
                    args.loss_gen   * g_loss_adv_det +
                    args.loss_quant * outputs['quantization_loss_det'] 
                )
                del g_loss_adv_det, generated_images_det, fake_score_det
                torch.cuda.empty_cache()
            # Compute regret
            regret = (g_loss_sample - g_loss_det).detach()

            # Retrieve log_probs from sample output
            # assume out_sample['log_probs'] is a tensor of same shape as batch, sum over appropriate dims
            B = batch['image'].size(0)
            N = outputs['log_probs'].numel() // B
            logp = outputs['log_probs'].view(B, N).mean(dim=1) # [B]
            # Policy gradient loss
            policy_loss = (regret * logp).mean()
            accelerator.backward(policy_loss)
            torch.nn.utils.clip_grad_norm_(generator.image_generation_model.weight_module.parameters(), max_norm=1.0)
            optimizer_W.step()

            total_policy_loss += policy_loss.item()
            total_regret += regret.mean().item()

            
            # Update progress bar on main process
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    'policy_loss': f"{policy_loss.item():.4f}",
                    'regret': f"{regret.mean().item():.4f}"
                })
                progress_bar.update(1)
            # Log batch statistics only on main process
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                with torch.no_grad():
                    visualize_batch(
                        batch['image'][0],
                        outputs['generated_images_det'],
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
                    "Batch %d stats - PolicyLoss=%.4f, Regret=%.4f",
                    batch_idx,
                    policy_loss.item(),
                    regret.item()
                )
    except Exception as e:
        print(f"Error in train_epoch: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
    
    
    return total_policy_loss / batch_idx, total_regret / batch_idx


def validate(generator, val_loader, epoch, device, args, accelerator, stage=1):
    """Validate the model and optionally store generated outputs."""
    total_loss = 0
    num_batches = 0

    save_path = args.generated_data_dir

    if args.store_gen_data:
        os.makedirs(args.generated_data_dir, exist_ok=True)
        save_path = os.path.join(args.generated_data_dir, f"generated_epoch_{epoch}.jsonl")

    with torch.no_grad():

        for batch_idx, batch in enumerate(val_loader):
            with autocast():
                outputs = generator(batch['image'], batch['question'], batch['answer_text'], stage=stage)

                loss = (
                    args.loss_recon * outputs['loss_recon'] +
                    args.loss_perc * outputs['loss_perc'] +
                    args.loss_vgg * outputs['loss_vgg']
                )
                total_loss += loss.item()
                num_batches += 1

                if args.store_gen_data:
                    for i in range(outputs['generated_images'].size(0)):
                        store_generated_outputs(
                            outputs['generated_images'][i],
                            batch['question'][i],
                            batch['answer_text'][i],
                            batch['image_id'][i].item() if torch.is_tensor(batch['image_id'][i]) else batch['image_id'][i],
                            save_path
                        )

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
    # Only set NCCL_SOCKET_IFNAME if we actually discovered one
    valid_interface = pick_real_interface()
    if valid_interface:
        os.environ["NCCL_SOCKET_IFNAME"] = valid_interface
    else:
        print("Warning: falling back to NCCL auto-detect")
    
    # Load configuration from config.json
    config = ModelConfig.from_json("config.json")
    config.override_with_args(args)
    print("Loaded Configuration:")
    
    # Get local rank for device placement
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(f'cuda:{local_rank}' if local_rank != -1 else 'cuda')
    
    # Initialize process group with TCP backend as fallback
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.distributed.init_process_group(
        backend="nccl",
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
        split_batches=False,           # don’t split batches across devices
    )
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="prompt_image_segment",
            dir=args.output_dir,
            config=args,
            name=run_name,
            resume="allow",
            mode="disabled"
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

    if torch.distributed.is_initialized():
        # Use DDP for discriminator instead of FSDP
        discriminator = DistributedDataParallel(
            discriminator,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Create separate optimizers
    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=args.learning_rate_g,  # Reduce learning rate
        betas=(0.5, 0.999)
    )
    
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate_d,  # Reduce learning rate
        betas=(0.5, 0.999)
    )

    # Create optimizer for RL_weight_agent only
    WM_params = [p for n, p in generator.named_parameters() if 'weight_module' in n and p.requires_grad]
    optimizer_W = torch.optim.Adam(
        WM_params,
        lr=args.learning_rate_w,
        betas=(0.5, 0.999)
    )
            

    # Create cosine annealing scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Keep initial learning rate during warmup
            return 1.0
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (args.num_epochs_1 - warmup_epochs)
            return 0.01 + (1 - 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)
    scheduler_W = torch.optim.lr_scheduler.LambdaLR(optimizer_W, lr_lambda)

    # Load checkpoint if it exists
    start_epoch = args.start_epoch
    best_loss = float('inf')
    if args.resume_generator_checkpoint and os.path.exists(args.resume_generator_checkpoint):
        load_checkpoint(generator, args.resume_generator_checkpoint, device)
        # Override loaded start_epoch if specified in args
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
    
    if args.resume_discriminator_checkpoint and os.path.exists(args.resume_discriminator_checkpoint):
        load_checkpoint(discriminator, args.resume_discriminator_checkpoint, device)
        # Override loaded start_epoch if specified in args
        if args.start_epoch > 0:
            start_epoch = args.start_epoch

    if args.resume_weight_checkpoint and os.path.exists(args.resume_weight_checkpoint):
        load_checkpoint(generator.image_generation_model.weight_module, args.resume_weight_checkpoint, device)
        # Override loaded start_epoch if specified in args
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
        

    
    # Prepare dataloaders
    train_dataloader, val_dataloader = create_vqa_dataloaders(args)
    
    # Let accelerator handle distribution
    generator, optimizer_G, optimizer_W, train_dataloader, val_dataloader = accelerator.prepare(
        generator, optimizer_G, optimizer_W, train_dataloader, val_dataloader
    )

    
    
    try:
        # Training loop
        if args.traditional:
            jpg = JPGTransmission(args, device=device)
            total_u_loss = 0
            num_batches = 0
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating JPG")):
                # Assume batch['image'] is a tensor of shape (B, C, H, W) with values in [0, 1]
                batch_size = batch['image'].shape[0] 
                num_batches += 1
                
                # Process each image in the batch individually
                for i in range(batch_size):
                    u_loss = jpg(batch['image'][i], batch['question'][i], batch['answer_text'][i], batch['image_id'][i])  # calls the forward() method of JPGTransmission
                    total_u_loss += u_loss.item()
            t_loss    = torch.tensor(total_u_loss,    dtype=torch.float32, device=device)
            n_batches = torch.tensor(num_batches,   dtype=torch.float32, device=device)
            # avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            torch.distributed.all_reduce(t_loss,    op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(n_batches, op=dist.ReduceOp.SUM)
            global_avg_loss = t_loss / n_batches
            if torch.distributed.get_rank() == 0:
                print(f"Aaverage loss: {global_avg_loss.item():.4f}")
            return global_avg_loss.item()
                
        if args.eval:
            total_loss = 0
            num_batches = 0

            save_path = args.generated_data_dir

            if args.store_gen_data:
                os.makedirs(args.generated_data_dir, exist_ok=True)
                save_path = args.generated_data_dir

            with torch.no_grad():
                val_iter = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating", dynamic_ncols=True)

                for batch_idx, batch in val_iter:
                    if batch_idx > 100:
                        break
                    with autocast():
                        outputs = generator(batch['image'], batch['question'], batch['answer_text'], stage=2)
                        total_loss += outputs['loss_perc'].item()
                        num_batches += 1

                        if args.store_gen_data:
                            for i in range(outputs['generated_images'].size(0)):
                                save_reconstructed_image(
                                    outputs['generated_images'][i],
                                    save_path,
                                    batch['image_id'][i].item() if torch.is_tensor(batch['image_id'][i]) else batch['image_id'][i],
                                )
            t_loss    = torch.tensor(total_loss,    dtype=torch.float32, device=device)
            n_batches = torch.tensor(num_batches,   dtype=torch.float32, device=device)
            # avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            torch.distributed.all_reduce(t_loss,    op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(n_batches, op=dist.ReduceOp.SUM)
            global_avg_loss = t_loss / n_batches
            if torch.distributed.get_rank() == 0:
                print(f"Aaverage loss: {global_avg_loss.item():.4f}")
            return global_avg_loss.item()


        if args.start_stage == 1:
            logger.info("Starting training stage 1: Encoder-Decoder")
            num_epochs_phase_1 = args.num_epochs_1/2

            for epoch in range(start_epoch, args.num_epochs_1):
                if epoch == start_epoch:
                    logger.info("Starting training in phase I")
                if epoch == num_epochs_phase_1:
                    logger.info("Starting training in phase II")
            
                with autocast():
                    generator.train()
                    if epoch < num_epochs_phase_1:
                        train_g_loss, train_q_loss = stage1_train(generator, discriminator, train_dataloader, [optimizer_G, optimizer_D], epoch, device, args, accelerator, phase=1)
                        # Run validation every 10 epochs
                        if epoch % 10 == 0:
                            generator.eval()
                            val_loss = validate(generator, val_dataloader, epoch, device, args, accelerator, stage=1)
                            logger.info(f"Epoch {epoch} - Train G loss: {train_g_loss:.4f}, Val loss: {val_loss:.4f}")
                        scheduler_G.step()
                        # Log learning rate
                        if accelerator.is_main_process:
                            current_lr_G = optimizer_G.param_groups[0]['lr']
                            wandb.log({
                                'epoch': epoch,
                                'learning_rate_G': current_lr_G,
                                'epoch_train_g_loss': train_g_loss,
                                'epoch_train_q_loss': train_q_loss,
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
                    else:
                        train_g_loss, train_a_loss, train_d_loss, train_q_loss, train_p_loss = stage1_train(generator, discriminator, train_dataloader, [optimizer_G, optimizer_D], epoch, device, args, accelerator, phase=2)
                        # Run validation every 10 epochs
                        if epoch % 10 == 0:
                            generator.eval()
                            val_loss = validate(generator, val_dataloader, epoch, device, args, accelerator, stage=1)
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
                                'epoch_train_q_loss': train_q_loss,
                                'epoch_train_p_loss': train_p_loss,
                                'epoch_val_loss': val_loss,
                            })
                            
                            # Save checkpoint asynchronously
                            logger.info(f"Saving generator checkpoint for epoch {epoch} with val_loss: {val_loss}")
                            gen_save_thread = threading.Thread(
                                target=save_checkpoint,
                                args=(
                                    generator.module if hasattr(generator, "module") else generator,
                                    'generator',
                                    epoch,
                                    val_loss,
                                    args
                                )
                            )

                            logger.info(f"Saving discriminator checkpoint for epoch {epoch} with val_loss: {val_loss}")
                            disc_save_thread = threading.Thread(
                                target=save_checkpoint,
                                args=(
                                    discriminator.module if hasattr(discriminator, "module") else discriminator,
                                    'discriminator',
                                    epoch,
                                    val_loss,
                                    args
                                )
                            )

                            gen_save_thread.start()
                            logger.info(f"generator thread alive after start? {gen_save_thread.is_alive()}")
                            disc_save_thread.start()
                            logger.info(f"discriminator thread alive after start? {disc_save_thread.is_alive()}")
                            
                            # Wait for saving to complete before synchronization
                            gen_save_thread.join()
                            disc_save_thread.join()
                            logger.info(f"Checkpoint saving completed")
                  
                torch.cuda.empty_cache()  # Clear memory before synchronization
                
                # All processes wait here once per epoch with longer timeout
                accelerator.wait_for_everyone() 
        elif args.start_stage == 2:
            logger.info("Starting training stage 2: Semantic-aware HBF")
            if args.apply_weight==1:
                generator.train()
                for p in generator.image_generation_model.weight_module.parameters():
                    p.requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = False

                for epoch in range(start_epoch, args.num_epochs_2):
                    with autocast():
                        generator.train()
                        total_policy_loss, total_regret = train_weight_module(generator, discriminator, train_dataloader, optimizer_W, epoch, device, args, accelerator)
                        # Run validation every 10 epochs
                        val_loss = 0.0
                        # if epoch % 10 == 0:
                        #     generator.zero_grad(set_to_none=True)
                        #     generator.eval()
                        #     val_loss = validate(generator, val_dataloader, epoch, device, args, accelerator, stage=2)
                        #     logger.info(f"Epoch {epoch} - total_policy_loss: {total_policy_loss:.4f}, total_regret: {total_regret:.4f}")
                        scheduler_W.step()
                        if accelerator.is_main_process:
                            current_lr_W = optimizer_W.param_groups[0]['lr']
                            wandb.log({
                                'epoch': epoch,
                                'learning_rate_W': current_lr_W,
                                'total_policy_loss': total_policy_loss,
                                'total_regret': total_regret,
                                'epoch_val_loss': val_loss,
                            })
                            
                            # Save checkpoint asynchronously
                            logger.info(f"Saving weight checkpoint for epoch {epoch} with val_loss: {val_loss}")
                            save_thread = threading.Thread(
                                target=save_checkpoint,
                                args=(
                                    generator.image_generation_model.weight_module.module if hasattr(generator.image_generation_model.weight_module, "module") else generator.image_generation_model.weight_module,
                                    'weight_module',
                                    epoch,
                                    val_loss,
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

            else:
                for epoch in range(start_epoch, args.num_epochs_2):
                    with autocast():
                        generator.train()
                        train_g_loss, train_a_loss, train_d_loss, train_q_loss, train_p_loss = stage2_train(generator, discriminator, train_dataloader, [optimizer_G, optimizer_D], epoch, device, args, accelerator)
                        # Run validation every 10 epochs
                        if epoch % 10 == 0:
                            generator.eval()
                            val_loss = validate(generator, val_dataloader, epoch, device, args, accelerator, stage=2)
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
                                'epoch_train_q_loss': train_q_loss,
                                'epoch_train_p_loss': train_p_loss,
                                'epoch_val_loss': val_loss,
                            })
                            
                            # Save checkpoint asynchronously
                            logger.info(f"Saving generator checkpoint for epoch {epoch} with val_loss: {val_loss}")
                            gen_save_thread = threading.Thread(
                                target=save_checkpoint,
                                args=(
                                    generator.module if hasattr(generator, "module") else generator,
                                    'generator',
                                    epoch,
                                    val_loss,
                                    args
                                )
                            )

                            logger.info(f"Saving discriminator checkpoint for epoch {epoch} with val_loss: {val_loss}")
                            disc_save_thread = threading.Thread(
                                target=save_checkpoint,
                                args=(
                                    discriminator.module if hasattr(discriminator, "module") else discriminator,
                                    'discriminator',
                                    epoch,
                                    val_loss,
                                    args
                                )
                            )

                            gen_save_thread.start()
                            disc_save_thread.start()
                            
                            # Wait for saving to complete before synchronization
                            gen_save_thread.join()
                            disc_save_thread.join()
                            logger.info(f"Rank 0: Checkpoint saving completed")
                    
                torch.cuda.empty_cache()  # Clear memory before synchronization
                
                # All processes wait here once per epoch with longer timeout
                accelerator.wait_for_everyone() 


        elif args.start_stage == 3:
            logger.info("Starting training stage 3: Full fine-tuning")
            for epoch in range(start_epoch, args.num_epochs):
                generator.train()
                train_g_loss, train_a_loss, train_d_loss, train_q_loss = train_epoch(generator, discriminator, train_dataloader, [optimizer_G, optimizer_D], epoch, device, args, accelerator)
                
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
                        'epoch_train_q_loss': train_q_loss,
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

                
                torch.cuda.empty_cache()
                # All processes wait here once per epoch with longer timeout
                accelerator.wait_for_everyone()
        else:
            logger.info("Training stage not specified. Exiting.")

                
            
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