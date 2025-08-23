#!/usr/bin/env python3
"""Fine-tuning script for math problem solving."""

import os
import sys
import argparse
import logging
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.transformer import TransformerForCausalLM
from src.data.math_dataset import create_math_dataloaders
from src.training.math_trainer import MathFineTuningTrainer

import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config: dict):
    """Create model configuration object."""
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return ModelConfig(config['model'])


def main():
    """Main fine-tuning entry point."""
    parser = argparse.ArgumentParser(description='Fine-tune LLM for math problem solving')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/finetune_config.yaml',
        help='Path to fine-tuning configuration file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use from the dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default=None,
        help='Path to pre-trained model checkpoint (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override pretrained model path if provided
    if args.pretrained_model:
        config['infrastructure']['pretrained_model_path'] = args.pretrained_model
    
    # Check if pre-trained model path exists
    pretrained_path = config['infrastructure'].get('pretrained_model_path')
    if not pretrained_path or not Path(pretrained_path).exists():
        logger.error(f"Pre-trained model path not found: {pretrained_path}")
        logger.error("Please provide a valid pre-trained model checkpoint")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Starting LLM Fine-tuning for Math Problem Solving")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Pre-trained model: {pretrained_path}")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config['infrastructure']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['infrastructure']['seed'])
    
    # Create tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info("Creating model...")
    model_config = create_model_config(config)
    model = TransformerForCausalLM(model_config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_math_dataloaders(
        config=config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )
    
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataloader.dataset)}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MathFineTuningTrainer(
        model=model,
        config=config,
        device=device,
    )
    
    # Load resume checkpoint if specified
    if config['infrastructure']['resume_from_checkpoint']:
        trainer.load_checkpoint(config['infrastructure']['resume_from_checkpoint'])
    
    # Training loop
    logger.info("Starting fine-tuning...")
    epoch = trainer.epoch
    
    try:
        while trainer.global_step < config['training']['num_training_steps']:
            logger.info(f"Starting epoch {epoch}")
            trainer.train_epoch(train_dataloader, eval_dataloader, epoch)
            epoch += 1
            trainer.epoch = epoch
            
            # Check if training completed
            if trainer.global_step >= config['training']['num_training_steps']:
                break
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Final checkpoint and evaluation
    logger.info("Saving final checkpoint...")
    trainer.save_checkpoint()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate(eval_dataloader)
    logger.info("Final evaluation results:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("=" * 60)
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Best model saved at: {Path(config['infrastructure']['output_dir']) / 'best_model'}")
    logger.info(f"Final checkpoint saved at: {Path(config['infrastructure']['output_dir']) / f'checkpoint-{trainer.global_step}'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()