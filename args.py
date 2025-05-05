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
    parser.add_argument("--learning_rate_w", type=float, default=5e-5,
                       help="Initial learning rate for discriminator")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW")
    parser.add_argument("--num_epochs_1", type=int, default=10,
                       help="Number of training epochs for stage 1")
    parser.add_argument("--num_epochs_2", type=int, default=10,
                       help="Number of training epochs for stage 1")
    parser.add_argument("--num_epochs_3", type=int, default=10,
                       help="Number of training epochs for stage 1")
    parser.add_argument("--start_epoch", type=int, default=0,
                       help="Starting epoch number (useful for resuming training)")
    parser.add_argument("--start_stage", type=int, choices=[1, 2, 3], default=1,
                       help="Starting training stage: 1 (encoder-decoder), 2 (semantic-aware HBF), 3 (full fine-tuning)")
    parser.add_argument("--textalign", action="store_true", 
                       help="Enable text alignment")
    parser.add_argument("--apply_weight", type=int, default=0, choices=[0,1],
                       help="Apply weight to the SA-HBF")
    parser.add_argument("--store_gen_data", action="store_true", 
                       help="Whether to store generator data")
    parser.add_argument("--traditional", action="store_true", 
                       help="encode/decode by traditional methods, e.g. jpg")
    parser.add_argument("--eval", action="store_true", 
                       help="Run validation only")
    
    # Dataset and model arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--generated_data_dir", type=str, default="generated data dir")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to VQA v2 dataset')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume_generator_checkpoint', type=str, default=None)
    parser.add_argument('--resume_discriminator_checkpoint', type=str, default=None)
    parser.add_argument('--resume_weight_checkpoint', type=str, default=None)
    
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
    parser.add_argument('--loss_quant', type=float, default=0.1, 
                       help='Weight for quantization loss')
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
    parser.add_argument('--train_category', type=str, choices=['nonanimal', 'nonhuman', 'indoor', 'outdoor', 'None'],
                        help='Category to use for training. Must be one of: nonanimal, nonhuman, indoor, outdoor')
    parser.add_argument('--val_category', type=str, choices=['animal', 'human', 'indoor', 'outdoor', 'None'],
                        help='Category to use for validation. Must be one of: animal, human, indoor, outdoor')


    # --- Physical Layer Overrides (optional, overrides config.json if used) ---
    parser.add_argument("--Nt", type=int, default=None, help="Override number of transmit antennas")
    parser.add_argument("--Nr", type=int, default=None, help="Override number of receive antennas")
    parser.add_argument("--NRF", type=int, default=None, help="Override number of RF chains")
    parser.add_argument("--Ns", type=int, default=None, help="Override number of data streams")
    parser.add_argument("--num_subcarriers", type=int, default=None, help="Override number of OFDM subcarriers")
    parser.add_argument("--SNR", type=float, default=None, help="Override noise power")
    parser.add_argument("--M", type=int, default=None, help="Override modulation order")

    # --- Codebook Overrides (optional, overrides config.json if used) ---
    parser.add_argument("--num_embeddings", type=int, default=None, help="Override number of codewords")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Override dimension of codewords")

    # --- Channel Parameter Overrides ---
    parser.add_argument("--num_clusters", type=int, default=None, help="Override number of channel clusters")
    parser.add_argument("--num_rays", type=int, default=None, help="Override number of rays per cluster")
    
    return parser.parse_args()