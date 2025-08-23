"""Math problem solving inference script."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer
import logging
import argparse
import json
from pathlib import Path
import sys
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.transformer import TransformerForCausalLM

logger = logging.getLogger(__name__)


class MathProblemSolver:
    """Math problem solver using fine-tuned model."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Conversation template
        self.conversation_template = "Human: {problem}\n\nAssistant: "
        
    @torch.no_grad()
    def solve_problem(
        self,
        problem: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Solve a math problem and return the solution."""
        
        # Format the problem using conversation template
        formatted_input = self.conversation_template.format(problem=problem.strip())
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            truncation=True,
            max_length=max_length // 2,  # Leave room for generation
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Generate solution
        generated_ids = input_ids.clone()
        generated_attention_mask = attention_mask.clone()
        
        # Default stop tokens
        if stop_tokens is None:
            stop_tokens = ["Human:", "\n\nHuman:", "Human ", "\n\nHuman "]
        
        stop_token_ids = [
            self.tokenizer.encode(token, add_special_tokens=False)
            for token in stop_tokens
        ]
        
        # Generation loop
        for step in range(max_length - input_ids.shape[1]):
            # Get model predictions
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
            )
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
                    
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
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
                
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            
            # Update attention mask
            generated_attention_mask = torch.cat([
                generated_attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
            
            # Check for stop tokens
            if self._should_stop(generated_ids[0], stop_token_ids):
                break
                
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Decode the generated solution
        full_generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Extract just the assistant's response
        assistant_response = self._extract_assistant_response(full_generated_text, formatted_input)
        
        # Extract the final answer if present
        final_answer = self._extract_final_answer(assistant_response)
        
        return {
            'problem': problem,
            'solution': assistant_response,
            'final_answer': final_answer,
            'full_response': full_generated_text,
        }
        
    def _should_stop(self, generated_ids: torch.Tensor, stop_token_ids: List[List[int]]) -> bool:
        """Check if generation should stop based on stop tokens."""
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        for stop_tokens in stop_token_ids:
            if len(stop_tokens) > 0:
                stop_text = self.tokenizer.decode(stop_tokens, skip_special_tokens=True)
                if stop_text in generated_text:
                    return True
        return False
        
    def _extract_assistant_response(self, full_text: str, input_text: str) -> str:
        """Extract just the assistant's response from the full generated text."""
        # Remove the input prompt
        if input_text in full_text:
            response = full_text[len(input_text):].strip()
        else:
            # Fallback: look for "Assistant:" and extract everything after
            if "Assistant:" in full_text:
                response = full_text.split("Assistant:", 1)[1].strip()
            else:
                response = full_text.strip()
                
        # Remove any trailing stop tokens
        stop_patterns = ["Human:", "\n\nHuman:", "Human ", "\n\nHuman "]
        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()
                
        return response
        
    def _extract_final_answer(self, solution: str) -> Optional[str]:
        """Extract the final answer from the solution using common patterns."""
        # Common patterns for final answers in math problems
        patterns = [
            r'\\boxed\{([^}]+)\}',  # LaTeX boxed answers
            r'\\boxed\{([^}]*)\}',  # LaTeX boxed (empty allowed)
            r'Therefore,?\s*([^.]+)\.',  # "Therefore, X."
            r'Thus,?\s*([^.]+)\.',  # "Thus, X."
            r'The answer is:?\s*([^.]+)',  # "The answer is X"
            r'Final answer:?\s*([^.]+)',  # "Final answer: X"
            r'Answer:?\s*([^.]+)',  # "Answer: X"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            if matches:
                return matches[-1].strip()  # Return the last match
                
        return None
        
    def solve_batch(
        self,
        problems: List[str],
        **generation_kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        """Solve multiple problems."""
        results = []
        for i, problem in enumerate(problems):
            print(f"Solving problem {i+1}/{len(problems)}...")
            result = self.solve_problem(problem, **generation_kwargs)
            results.append(result)
        return results


def load_math_solver(checkpoint_path: str, device: str = 'cuda') -> MathProblemSolver:
    """Load the fine-tuned model and create solver."""
    
    # Load config
    config_path = Path(checkpoint_path) / 'finetune_config.json'
    if not config_path.exists():
        # Fallback to original config
        config_path = Path(checkpoint_path) / 'config.json'
        
    with open(config_path, 'r') as f:
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
        
    # Create solver
    solver = MathProblemSolver(model, tokenizer, device)
    
    return solver


def main():
    """Interactive math problem solver."""
    parser = argparse.ArgumentParser(description='Solve math problems using fine-tuned LLM')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to fine-tuned model checkpoint'
    )
    parser.add_argument(
        '--problem',
        type=str,
        help='Math problem to solve (if not provided, will start interactive mode)'
    )
    parser.add_argument(
        '--problems-file',
        type=str,
        help='File containing multiple problems (one per line)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1024,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k filtering'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) filtering'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Load solver
    print("Loading math problem solver...")
    solver = load_math_solver(args.checkpoint, args.device)
    print("✓ Solver loaded successfully!")
    
    generation_kwargs = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }
    
    results = []
    
    if args.problems_file:
        # Batch mode: solve problems from file
        print(f"Loading problems from {args.problems_file}...")
        with open(args.problems_file, 'r') as f:
            problems = [line.strip() for line in f if line.strip()]
        
        print(f"Solving {len(problems)} problems...")
        results = solver.solve_batch(problems, **generation_kwargs)
        
        # Print results
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"Problem {i+1}: {result['problem']}")
            print(f"\nSolution:\n{result['solution']}")
            if result['final_answer']:
                print(f"\nFinal Answer: {result['final_answer']}")
            print('='*60)
            
    elif args.problem:
        # Single problem mode
        print(f"Problem: {args.problem}")
        print("Solving...")
        result = solver.solve_problem(args.problem, **generation_kwargs)
        results = [result]
        
        print(f"\nSolution:\n{result['solution']}")
        if result['final_answer']:
            print(f"\nFinal Answer: {result['final_answer']}")
            
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Math Problem Solver")
        print("Type 'quit' or 'exit' to stop")
        print("="*60)
        
        while True:
            try:
                problem = input("\nEnter a math problem: ").strip()
                if problem.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not problem:
                    continue
                    
                print("Solving...")
                result = solver.solve_problem(problem, **generation_kwargs)
                results.append(result)
                
                print(f"\nSolution:\n{result['solution']}")
                if result['final_answer']:
                    print(f"\nFinal Answer: {result['final_answer']}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    # Save results if output file specified
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()