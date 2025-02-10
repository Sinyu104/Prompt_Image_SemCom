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
    logger.setLevel(logging.INFO)
    
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

def save_checkpoint(model, optimizer, epoch, loss, args):
    """Save model checkpoint."""
    logger = logging.getLogger('training')
    try:
        logger.info("Saving checkpoint...")
        
        model_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_state_dict[name] = param.cpu()
        
        logger.info("Number of parameters being saved: %d", len(model_state_dict))
        
        optimizer_state_dict = {
            k: v for k, v in optimizer.state_dict().items()
            if 'state' not in k or any(p.requires_grad for p in model.parameters())
        }
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
        }
        
        checkpoint_path = os.path.join(
            args.output_dir, 
            f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        
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
        
        for key, v in checkpoint[state_dict_key].items():
            if key in current_state:
                current_state[key] = v
            else:
                logger.warning(f"Warning: Checkpoint parameter {key} not found in current model")
        
        model.load_state_dict(current_state)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            logger.info("Loading optimizer state...")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        logger.info(f"Loaded {len(checkpoint[state_dict_key])} trainable parameters")
        
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

def visualize_batch(image, generated_images, question, gt_answer, epoch, batch_idx, args, mode='train'):
    """Visualize a batch of images and save to wandb."""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:  # If channels first, transpose
        image = np.transpose(image, (1, 2, 0))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Generated image
    if isinstance(generated_images, torch.Tensor):
        generated_image = generated_images[0].cpu().numpy()
    else:
        generated_image = generated_images[0]
    if generated_image.shape[0] == 3:
        generated_image = np.transpose(generated_image, (1, 2, 0))
    ax2.imshow(generated_image)
    ax2.set_title('Generated Image')
    ax2.axis('off')
    
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
        'question': question,
        'ground_truth_answer': gt_answer
    })
    
    plt.close() 