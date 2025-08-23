"""Fine-tuning trainer for math problem solving."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class MathFineTuningTrainer:
    """Fine-tuning trainer for math problem solving."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda',
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing
        if config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config['training']['fp16'] else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup logging
        self.writer = SummaryWriter(config['infrastructure']['logging_dir'])
        self._setup_logging()
        
        # Load pre-trained checkpoint if specified
        if config['infrastructure'].get('pretrained_model_path'):
            self.load_pretrained_checkpoint(config['infrastructure']['pretrained_model_path'])
            
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=(
                self.config['training']['adam_beta1'],
                self.config['training']['adam_beta2']
            ),
            eps=self.config['training']['adam_epsilon']
        )
        
    def _create_scheduler(self):
        """Create cosine annealing scheduler."""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_training_steps'],
            eta_min=self.config['training']['min_learning_rate']
        )
        
    def _setup_logging(self):
        """Setup logging directories and files."""
        # Create directories
        Path(self.config['infrastructure']['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['infrastructure']['logging_dir']).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = Path(self.config['infrastructure']['output_dir']) / 'finetune_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pre-trained model checkpoint."""
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_path}")
        
        try:
            # Load model state
            model_state = torch.load(
                Path(checkpoint_path) / 'pytorch_model.bin',
                map_location=self.device
            )
            self.model.load_state_dict(model_state, strict=True)
            logger.info("Successfully loaded pre-trained model weights")
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained checkpoint: {e}")
            raise
            
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss / self.config['training']['gradient_accumulation_steps']
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / self.config['training']['gradient_accumulation_steps']
            
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return {'loss': loss.item() * self.config['training']['gradient_accumulation_steps']}
        
    def evaluate(self, dataloader) -> Dict:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_samples': total_samples
        }
        
    def train_epoch(self, train_dataloader, eval_dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
        )
        
        for step, batch in enumerate(progress_bar):
            # Training step
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            
            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    avg_loss = total_loss / (step + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step
                    })
                    
                # Evaluation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    # Log evaluation metrics
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{key}', value, self.global_step)
                        
                    logger.info(
                        f"Step {self.global_step}: "
                        f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                        f"eval_perplexity={eval_metrics['eval_perplexity']:.2f}"
                    )
                    
                    # Save best model
                    if (self.config['evaluation']['save_best_model'] and 
                        eval_metrics['eval_loss'] < self.best_eval_loss):
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(is_best=True)
                        logger.info(f"New best model saved with eval_loss={self.best_eval_loss:.4f}")
                    
                    self.model.train()
                    
                # Regular checkpointing
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()
                    
                # Check if we've reached max steps
                if self.global_step >= self.config['training']['num_training_steps']:
                    return
                    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            checkpoint_dir = Path(self.config['infrastructure']['output_dir']) / 'best_model'
        else:
            checkpoint_dir = Path(self.config['infrastructure']['output_dir']) / f'checkpoint-{self.global_step}'
            
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / 'pytorch_model.bin'
        )
        
        # Save optimizer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }, checkpoint_dir / 'training_state.pt')
        
        # Save config
        with open(checkpoint_dir / 'finetune_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Clean up old checkpoints (keep only the latest N)
        if not is_best:
            self._cleanup_checkpoints()
            
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the latest N."""
        checkpoint_dirs = sorted(
            Path(self.config['infrastructure']['output_dir']).glob('checkpoint-*'),
            key=lambda x: int(x.name.split('-')[1])
        )
        
        if len(checkpoint_dirs) > self.config['training']['save_total_limit']:
            for checkpoint_dir in checkpoint_dirs[:-self.config['training']['save_total_limit']]:
                import shutil
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Removed old checkpoint: {checkpoint_dir}")
                
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming fine-tuning."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model state
        model_state = torch.load(
            Path(checkpoint_path) / 'pytorch_model.bin',
            map_location=self.device
        )
        self.model.load_state_dict(model_state)
        
        # Load training state
        training_state = torch.load(
            Path(checkpoint_path) / 'training_state.pt',
            map_location=self.device
        )
        
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.best_eval_loss = training_state.get('best_eval_loss', float('inf'))
        
        if self.scaler is not None and training_state['scaler'] is not None:
            self.scaler.load_state_dict(training_state['scaler'])
            
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")