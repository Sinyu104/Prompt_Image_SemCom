import torch
import os
import logging
import netifaces
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast

import os
import json
import base64
from torchvision.utils import save_image
from io import BytesIO
from PIL import Image

def setup_logger(args):
    """Setup logger with file and console handlers"""
    # Only setup logger on the main process
    if int(os.environ.get("LOCAL_RANK", -1)) != 0:
        logger = logging.getLogger('training')
        logger.addHandler(logging.NullHandler())
        return logger
    
    # Create logger
    logger = logging.getLogger('training')
    logger.handlers.clear()
    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    log_file = os.path.join(args.output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def print_network_info():
    """Print available network interfaces for debugging"""
    print("\nAvailable Network Interfaces:")
    valid_interfaces = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            addr = addrs[netifaces.AF_INET][0]
            if 'addr' in addr:
                valid_interfaces.append(iface)
                print(f"Interface {iface}: {addr['addr']}")
    return valid_interfaces[0] if valid_interfaces else None

def print_memory_stats():
    logger = logging.getLogger('training')
    if torch.cuda.is_available():
        logger.info("GPU Memory Stats:")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            logger.info(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    return ""

def save_checkpoint(model, epoch, loss, args):
    """Save model checkpoint and keep only the latest two."""
    logger = logging.getLogger('training')
    try:
        logger.info("Saving checkpoint...")
        
        model_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_state_dict[name] = param.cpu()
        
        logger.info("Number of parameters being saved: %d", len(model_state_dict))
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'loss': loss,
        }

        save_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            save_dir,
            f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Successfully saved checkpoint to {checkpoint_path}")
        
        # Keep only the latest two checkpoints
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint')]
        for ckpt in checkpoints:
            ckpt_epoch = int(ckpt.split('_')[2])
            if ckpt_epoch <= epoch - 2:
                os.remove(os.path.join(save_dir, ckpt))
                logger.info(f"Removed old checkpoint {ckpt}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint."""
    logger = logging.getLogger('training')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        logger.info(f"Loading checkpoint from {checkpoint_path} - start_epoch: {start_epoch}, best_loss: {best_loss}")
        
        current_state = model.state_dict()
        
        if 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'
        else:
            raise KeyError("No model state dict found in checkpoint")
        
        # Load only matching parameters
        matched_state_dict = {}
        for key, v in checkpoint[state_dict_key].items():
            if key in current_state:
                if current_state[key].shape == v.shape:
                    matched_state_dict[key] = v
                else:
                    logger.warning(f"Skipping parameter {key} due to shape mismatch: "
                                 f"checkpoint={v.shape}, model={current_state[key].shape}")
            else:
                logger.warning(f"Parameter {key} not found in current model")
        
        model.load_state_dict(matched_state_dict, strict=False)
        
        # Skip loading optimizer state when architecture changes
        logger.info("Skipping optimizer state due to architecture change")
            
        logger.info(f"Loaded {len(matched_state_dict)} matching parameters")
        
        return start_epoch, best_loss
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, float('inf')

def setup_distributed_training(args):
    """Setup distributed training"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    return world_size, local_rank

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()



def store_generated_outputs(image_tensor, question, answer, image_id, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to PIL and encode
    image = image_tensor.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Save as JSONL
    entry = {
        "image_id": image_id,
        "question": question,
        "answer": answer,
        "image_base64": img_str
    }
    with open(save_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def save_reconstructed_image(reconstructed, save_path, image_id):
    """
    Save the reconstructed image as a standalone file.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor (C, H, W) in [0, 1].
        save_path (str): Directory to save the image.
        image_id (int or str): Identifier for the image.
    """
    # Convert tensor to (H, W, C) numpy array
    reconstructed_np = reconstructed.detach().cpu().permute(1, 2, 0).numpy()

    # Clip values to [0, 1] for valid RGB display
    reconstructed_np = np.clip(reconstructed_np, 0.0, 1.0)

    # Plot and save
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstructed_np)
    plt.axis("off")
    plt.tight_layout()

    save_path_full = os.path.join(save_path, f"reconstructed_{image_id}.png")
    plt.savefig(save_path_full, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
        
@torch.no_grad()
def visualize_batch(image, generated_images, question, gt_answer, epoch, batch_idx, args, mode='train'):
    """Visualize a batch of images and save to wandb."""
    logger = logging.getLogger('training')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        logger.debug(f"Original image tensor stats - min: {image.min():.3f}, max: {image.max():.3f}, mean: {image.mean():.3f}")

    if image.shape[0] == 3:  # If channels first
        image = np.transpose(image, (1, 2, 0))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Generated image
    if isinstance(generated_images, torch.Tensor):
        generated_image = generated_images[0].cpu().numpy()
        logger.debug(f"Generated image tensor stats - min: {generated_image.min():.3f}, max: {generated_image.max():.3f}, mean: {generated_image.mean():.3f}")
    else:
        generated_image = generated_images[0]
        
    if generated_image.shape[0] == 3:
        generated_image = np.transpose(generated_image, (1, 2, 0))
        # Clamp values to [0,1] range for display
        generated_image = np.clip(generated_image, 0.0, 1.0)
        logger.debug(f"Generated image after transpose and clamp - shape: {generated_image.shape}, min: {generated_image.min():.3f}, max: {generated_image.max():.3f}")

    ax2.imshow(generated_image)
    ax2.set_title('Generated Image')
    ax2.axis('off')
    
    # Log color channel statistics
    logger.debug(f"Original image channel stats:")
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = image[..., i]
        logger.debug(f"{channel} - min: {channel_data.min():.3f}, max: {channel_data.max():.3f}, "
                    f"mean: {channel_data.mean():.3f}, std: {channel_data.std():.3f}")
    
    logger.debug(f"Generated image channel stats:")
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = generated_image[..., i]
        logger.debug(f"{channel} - min: {channel_data.min():.3f}, max: {channel_data.max():.3f}, "
                    f"mean: {channel_data.mean():.3f}, std: {channel_data.std():.3f}")
    
    # Also log overall image statistics
    logger.debug(f"Overall image stats:")
    logger.debug(f"Original - mean: {image.mean():.3f}, std: {image.std():.3f}")
    logger.debug(f"Generated - mean: {generated_image.mean():.3f}, std: {generated_image.std():.3f}")
    
    # Add question and ground truth answer as text
    plt.figtext(0.02, 0.02, f'Q: {question}\nGT: {gt_answer}', 
               wrap=True, horizontalalignment='left', fontsize=16)
    
    # Save plot locally with separate directories for train and val
    save_dir = os.path.join(args.output_dir, 'visualizations', mode, f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}.png'), 
                bbox_inches='tight', dpi=300)
    
    # Log to wandb with mode prefix
    wandb.log({
        f'{mode}_visualizations': wandb.Image(plt),
        'epoch': epoch,
        'batch': batch_idx,
    })
    
    plt.close()