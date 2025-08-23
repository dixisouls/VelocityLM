"""Dataset module for fine-tuning on NuminaMath-CoT dataset."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import json
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MathConversationDataset(Dataset):
    """Dataset for math conversation fine-tuning with NuminaMath-CoT."""
    
    def __init__(
        self,
        config: Dict,
        tokenizer: AutoTokenizer,
        max_samples: Optional[int] = None,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_seq_length = config['data']['max_seq_length']
        
        logger.info(f"Loading NuminaMath-CoT dataset (split: {split})...")
        
        # Load the dataset
        self.dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            streaming=False,  # Load fully for fine-tuning
        )
        
        # Limit samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        logger.info(f"Loaded {len(self.dataset)} samples for {split}")
        
        # Cache processed samples
        self.processed_samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        """Process and cache all samples."""
        logger.info("Processing conversation samples...")
        
        for idx, sample in enumerate(self.dataset):
            try:
                # Parse the messages column
                if isinstance(sample['messages'], str):
                    messages = json.loads(sample['messages'])
                else:
                    messages = sample['messages']
                
                # Convert conversation to training format
                processed_sample = self._process_conversation(messages)
                if processed_sample is not None:
                    self.processed_samples.append(processed_sample)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
                
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.dataset)} samples")
                
        logger.info(f"Successfully processed {len(self.processed_samples)} samples")
        
    def _process_conversation(self, messages: List[Dict]) -> Optional[Dict]:
        """Convert conversation messages to training format."""
        if not messages or len(messages) < 2:
            return None
            
        # Build conversation text
        conversation_text = ""
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                conversation_text += f"Human: {content}\n\n"
            elif role == 'assistant':
                conversation_text += f"Assistant: {content}\n\n"
                
        # Tokenize the conversation
        tokens = self.tokenizer(
            conversation_text.strip(),
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Optionally mask the human part in labels (only train on assistant responses)
        if self.config['training'].get('mask_human_tokens', False):
            labels = self._mask_human_tokens(conversation_text, labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
    def _mask_human_tokens(self, conversation_text: str, labels: torch.Tensor) -> torch.Tensor:
        """Mask human tokens in labels so model only learns from assistant responses."""
        # Find assistant response starts
        lines = conversation_text.split('\n')
        assistant_starts = []
        current_pos = 0
        
        for line in lines:
            if line.startswith('Assistant:'):
                # Find position in tokenized sequence
                prefix_text = conversation_text[:current_pos]
                prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)['input_ids']
                assistant_starts.append(len(prefix_tokens))
            current_pos += len(line) + 1  # +1 for newline
            
        # Mask everything except assistant responses
        masked_labels = labels.clone()
        masked_labels.fill_(-100)  # -100 is ignored in loss computation
        
        # Unmask assistant responses
        for start_idx in assistant_starts:
            if start_idx < len(masked_labels):
                # Find end of assistant response (next "Human:" or end)
                remaining_text = conversation_text[start_idx:]
                next_human = remaining_text.find('\n\nHuman:')
                if next_human != -1:
                    end_text = conversation_text[:start_idx + next_human]
                else:
                    end_text = conversation_text
                    
                end_tokens = self.tokenizer(end_text, add_special_tokens=False)['input_ids']
                end_idx = min(len(end_tokens), len(masked_labels))
                
                # Unmask assistant tokens
                masked_labels[start_idx:end_idx] = labels[start_idx:end_idx]
                
        return masked_labels
        
    def __len__(self):
        return len(self.processed_samples)
        
    def __getitem__(self, idx):
        return self.processed_samples[idx]


class MathDataCollator:
    """Data collator for math conversation fine-tuning."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch."""
        batch = {
            'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in examples]),
            'labels': torch.stack([ex['labels'] for ex in examples])
        }
        return batch


def create_math_dataloaders(
    config: Dict,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for math fine-tuning."""
    
    # Create datasets
    train_dataset = MathConversationDataset(
        config=config,
        tokenizer=tokenizer,
        max_samples=max_samples,
        split="train",
    )
    
    # Create validation dataset (using a subset of train for now)
    val_samples = min(1000, len(train_dataset) // 10) if max_samples is None else max_samples // 10
    val_dataset = MathConversationDataset(
        config=config,
        tokenizer=tokenizer,
        max_samples=val_samples,
        split="train",  # Using train split for now, adjust if validation split exists
    )
    
    # Create data collator
    collator = MathDataCollator(tokenizer=tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size_per_device'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size_per_device'],
        shuffle=False,
        collate_fn=collator,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader