"""Distributed trainer for LLM."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed trainer with mixed precision and gradient accumulation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        rank: int,
        world_size: int,
    ):
        self.model = model
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Move model to device
        self.device = torch.device(f'cuda:{rank}')
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing
        if config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
        
        # Wrap model in DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=config['distributed']['find_unused_parameters']
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config['training']['fp16'] else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Setup logging
        if self.is_main_process:
            self.writer = SummaryWriter(config['infrastructure']['logging_dir'])
            self._setup_logging()
            
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
        """Create cosine annealing scheduler with warmup."""
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['warmup_steps'],
            T_mult=1,
            eta_min=self.config['training']['min_learning_rate']
        )
        
    def _setup_logging(self):
        """Setup logging directories and files."""
        # Create directories
        Path(self.config['infrastructure']['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['infrastructure']['logging_dir']).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = Path(self.config['infrastructure']['output_dir']) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
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
        
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main_process
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
                    
                    if self.is_main_process:
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', lr, self.global_step)
                        
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step
                    })
                    
                # Checkpointing
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()
                    
                # Check if we've reached max steps
                if self.global_step >= self.config['training']['num_training_steps']:
                    return
                    
    def save_checkpoint(self):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
            
        checkpoint_dir = Path(self.config['infrastructure']['output_dir']) / f'checkpoint-{self.global_step}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(
            model_to_save.state_dict(),
            checkpoint_dir / 'pytorch_model.bin'
        )
        
        # Save optimizer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }, checkpoint_dir / 'training_state.pt')
        
        # Save config
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Clean up old checkpoints
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
        """Load checkpoint for resuming training."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model state
        model_state = torch.load(
            Path(checkpoint_path) / 'pytorch_model.bin',
            map_location=self.device
        )
        self.model.module.load_state_dict(model_state)
        
        # Load training state
        training_state = torch.load(
            Path(checkpoint_path) / 'training_state.pt',
            map_location=self.device
        )
        
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        
        if self.scaler is not None and training_state['scaler'] is not None:
            self.scaler.load_state_dict(training_state['scaler'])
            
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")