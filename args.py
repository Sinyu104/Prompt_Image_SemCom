import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for VQA with segmentation")
    
    # Add config file argument
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to config file (optional)")
    
    # Add training hyperparameters
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training mode")
    parser.add_argument("--learning_rate_g", type=float, default=5e-5,
                       help="Initial learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=5e-5,
                       help="Initial learning rate for discriminator")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                       help="Starting epoch number (useful for resuming training)")
    
    # Dataset and model arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to VQA v2 dataset')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    # Loss weights
    parser.add_argument('--lambda_sparsity', type=float, default=0.1)
    parser.add_argument('--lambda_smoothness', type=float, default=0.05)
    parser.add_argument('--lambda_answer', type=float, default=1.0,
                       help='Weight for answer relevance loss')
    parser.add_argument('--loss_recon', type=float, default=1.0, 
                       help='Weight for reconstruction loss')
    parser.add_argument('--loss_perc', type=float, default=0.1, 
                       help='Weight for perceptual loss')
    parser.add_argument('--loss_vgg', type=float, default=10.0, 
                       help='Weight for VGG perceptual loss')
    parser.add_argument('--loss_gen', type=float, default=0.1, help='Weight for generator adversarial loss')
    parser.add_argument('--loss_disc', type=float, default=1.0, help='Weight for discriminator loss')
    
    # Add discriminator update frequency argument
    parser.add_argument("--discriminator_update_freq", type=int, default=20,
                       help="Frequency of discriminator updates (in batches)")
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=100,
                       help='How many batches to wait before logging example predictions')
    parser.add_argument('--vis_interval', type=int, default=100, 
                       help='Interval for visualization')
    
    # Dataset splitting arguments (all optional)
    parser.add_argument('--split_type', type=str,
                       choices=['image_based', 'question_based'],
                       help='First level split type (optional)', default=None)
    parser.add_argument('--split_level2', type=str,
                       choices=['scene', 'subject', 'reasoning', 'type'],
                       help='Second level split category (optional)', default=None)
    parser.add_argument('--train_category', type=str,
                       help='Category to use for training (optional)', default=None)
    parser.add_argument('--val_category', type=str,
                       help='Category to use for validation (optional)', default=None)
    
    return parser.parse_args()