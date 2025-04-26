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
import glob

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

def save_checkpoint(model, model_name, epoch, loss, args):
    """
    Save only the trainable parameters of `model` under `model_name`,
    unwraps .module if present, and keeps only the 2 most recent checkpoints.
    """
    logger = logging.getLogger('training')
    # 1) unwrap if wrapped
    to_save = model.module if hasattr(model, "module") else model

    # 2) collect only trainable params
    trainable_state = {
        name: param.detach().cpu()
        for name, param in to_save.named_parameters()
        if param.requires_grad
    }
    logger.info(f"[{model_name}] trainable params: {len(trainable_state)} tensors")

    checkpoint = {
        "epoch":      epoch,
        "loss":       loss,
        "model_state_dict": trainable_state,
    }

    # 3) write out
    save_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_epoch{epoch}_loss{loss:.4f}.pth")
    torch.save(checkpoint, path)
    logger.info(f"[{model_name}] saved to {path}")

    # 4) prune old by modification time (keep only latest two)
    pattern = os.path.join(save_dir, f"{model_name}_epoch*.pth")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    for old in ckpts[:-2]:
        os.remove(old)
        logger.info(f"[{model_name}] removed old checkpoint {os.path.basename(old)}")




def load_checkpoint(model, checkpoint_path, device):
    """
    Load only the trainable parameters saved at `checkpoint_path` into `model`,
    unwraps .module if present, strips any 'module.' prefixes, and returns
    (start_epoch, best_loss).  Everything else remains frozen as in the model.
    """
    logger = logging.getLogger('training')
    # unwrap if wrapped
    base = model.module if hasattr(model, "module") else model

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    start_epoch = ckpt.get("epoch", 0)
    best_loss   = ckpt.get("loss", float("inf"))
    raw_state   = ckpt["model_state_dict"]

    # strip `module.` prefix if any
    fixed_state = { k.replace("module.", ""): v
                    for k, v in raw_state.items() }

    # load only these keys (strict=False will skip everything else)
    missing, unexpected = base.load_state_dict(fixed_state, strict=False)
    # get the set of parameter names you actually saved
    saved_param_keys = set(fixed_state.keys())

    # filter missing to only those that were supposed to come from your checkpoint
    missing_params = [k for k in missing if k in saved_param_keys]
    if missing_params:
        logger.warning(f"Missing tuned params: {missing_params}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

    logger.info(f"Loaded epoch {start_epoch}, best_loss={best_loss:.4f}")
    return start_epoch, best_loss

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