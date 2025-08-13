#!/usr/bin/env python3
"""Main training script for distributed LLM training."""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.transformer import TransformerForCausalLM
from src.model.layers import RMSNorm
from src.data.dataset import create_dataloaders
from src.training.trainer import DistributedTrainer

import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process group: rank={rank}, world_size={world_size}")


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config: Dict):
    """Create model configuration object."""
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return ModelConfig(config['model'])


def train_rank(rank: int, world_size: int, config: Dict, max_samples: int):
    """Training function for each rank."""
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Set seed
    torch.manual_seed(config['infrastructure']['seed'] + rank)
    torch.cuda.manual_seed_all(config['infrastructure']['seed'] + rank)
    
    # Create tokenizer
    logger.info(f"Rank {rank}: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info(f"Rank {rank}: Creating model...")
    model_config = create_model_config(config)
    model = TransformerForCausalLM(model_config)
    
    # Log model info
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create dataloader
    logger.info(f"Rank {rank}: Creating dataloader...")
    dataloader = create_dataloaders(
        config=config,
        tokenizer=tokenizer,
        max_samples=max_samples,
        rank=rank,
        world_size=world_size
    )
    
    # Create trainer
    logger.info(f"Rank {rank}: Creating trainer...")
    trainer = DistributedTrainer(
        model=model,
        config=config,
        rank=rank,
        world_size=world_size
    )
    
    # Load checkpoint if resuming
    if config['infrastructure']['resume_from_checkpoint']:
        trainer.load_checkpoint(config['infrastructure']['resume_from_checkpoint'])
    
    # Training loop
    logger.info(f"Rank {rank}: Starting training...")
    epoch = trainer.epoch
    
    while trainer.global_step < config['training']['num_training_steps']:
        logger.info(f"Rank {rank}: Starting epoch {epoch}")
        trainer.train_epoch(dataloader, epoch)
        epoch += 1
        trainer.epoch = epoch
        
        # Synchronize after each epoch
        dist.barrier()
    
    # Final checkpoint
    if rank == 0:
        trainer.save_checkpoint()
        logger.info("Training completed!")
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train LLM from scratch')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100000,
        help='Maximum number of samples to use from the dataset'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (default: all available)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine number of GPUs
    if args.num_gpus is not None:
        world_size = args.num_gpus
    else:
        world_size = torch.cuda.device_count()
        
    logger.info(f"Starting distributed training with {world_size} GPUs")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Launch distributed training
    mp.spawn(
        train_rank,
        args=(world_size, config, args.max_samples),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()