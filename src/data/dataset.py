"""Dataset module for streaming and processing RefinedWeb data."""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from typing import Dict, Optional, Iterator
from itertools import islice


logger = logging.getLogger(__name__)


class StreamingRefinedWebDataset(Dataset):
    """Streaming dataset for RefinedWeb with on-the-fly tokenization."""
    
    def __init__(
        self,
        config: Dict,
        tokenizer: AutoTokenizer,
        max_samples: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size
        self.max_seq_length = config['data']['max_seq_length']
        
        # Calculate samples per rank
        self.samples_per_rank = max_samples // world_size
        self.start_idx = rank * self.samples_per_rank
        self.end_idx = self.start_idx + self.samples_per_rank
        
        logger.info(f"Rank {rank}: Loading samples {self.start_idx} to {self.end_idx}")
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            config['data']['dataset_name'],
            config['data']['dataset_config'],
            split='train',
            streaming=True,
        )
        
        # Cache for processed samples
        self.cache = []
        self._prepare_cache()
        
    def _prepare_cache(self):
        """Pre-process and cache samples for this rank."""
        logger.info(f"Rank {self.rank}: Preparing cache...")
        
        # Skip to the starting index for this rank
        dataset_iter = iter(self.dataset)
        if self.start_idx > 0:
            dataset_iter = islice(dataset_iter, self.start_idx, None)
        
        count = 0
        for sample in dataset_iter:
            if count >= self.samples_per_rank:
                break
                
            text = sample[self.config['data']['text_column']]
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.cache.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze().clone()
            })
            
            count += 1
            if count % 1000 == 0:
                logger.info(f"Rank {self.rank}: Processed {count}/{self.samples_per_rank} samples")
                
        logger.info(f"Rank {self.rank}: Cache preparation complete. Total samples: {len(self.cache)}")
        
    def __len__(self):
        return len(self.cache)
        
    def __getitem__(self, idx):
        return self.cache[idx]


class DataCollatorForLanguageModeling:
    """Data collator for language modeling."""
    
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, examples):
        batch = {
            'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in examples]),
            'labels': torch.stack([ex['labels'] for ex in examples])
        }
        return batch


def create_dataloaders(
    config: Dict,
    tokenizer: AutoTokenizer,
    max_samples: int,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create distributed dataloaders for training."""
    
    # Create dataset
    dataset = StreamingRefinedWebDataset(
        config=config,
        tokenizer=tokenizer,
        max_samples=max_samples,
        rank=rank,
        world_size=world_size,
    )
    
    # Create data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size_per_device'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader