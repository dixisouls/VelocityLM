"""Text generation utilities for the trained model."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """Text generation with various decoding strategies."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate text from prompt(s)."""
        
        # Handle single string input
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
            )
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
                        
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample from the distribution
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device)
            ], dim=1)
            
            # Check for EOS token
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break
                
        # Decode generated sequences
        generated_texts = []
        for i in range(batch_size):
            generated_text = self.tokenizer.decode(
                generated_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(generated_text)
            
        return generated_texts
        
    def beam_search(
        self,
        prompt: str,
        max_length: int = 100,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> str:
        """Generate text using beam search."""
        # Implementation of beam search
        # This is a simplified version - full implementation would be more complex
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        # For now, fallback to greedy decoding
        return self.generate(
            prompt,
            max_length=max_length,
            do_sample=False,
            num_return_sequences=1
        )[0]


def load_generator(checkpoint_path: str, device: str = 'cuda'):
    """Load model and create generator."""
    import yaml
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.model.transformer import TransformerForCausalLM
    
    # Load config
    config_path = Path(checkpoint_path) / 'config.json'
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
        
    # Create model config
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
                
    model_config = ModelConfig(config['model'])
    
    # Load model
    model = TransformerForCausalLM(model_config)
    state_dict = torch.load(
        Path(checkpoint_path) / 'pytorch_model.bin',
        map_location=device
    )
    model.load_state_dict(state_dict)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Create generator
    generator = TextGenerator(model, tokenizer, device)
    
    return generator


if __name__ == '__main__':
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k filtering')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) filtering')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load generator
    print("Loading model...")
    generator = load_generator(args.checkpoint, args.device)
    
    # Generate text
    print(f"Prompt: {args.prompt}")
    print("Generating...")
    
    generated = generator.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    print(f"Generated: {generated[0]}")