"""Gradio app for the custom LLM with streaming support and ZeroGPU integration."""

import gradio as gr
import torch
import torch.nn.functional as F
from typing import Iterator, Optional, Union, List
from transformers import AutoTokenizer
import json
import warnings
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings("ignore")

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Mock decorator for local testing
    def spaces_decorator(gpu_memory=None):
        def decorator(func):
            return func
        return decorator
    spaces = type('MockSpaces', (), {'GPU': spaces_decorator})

from src.model.transformer import TransformerForCausalLM


class StreamingTextGenerator:
    """Streaming text generation for the custom LLM."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: Optional[int] = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Iterator[str]:
        """Generate text with streaming output."""
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=1024,  # Leave room for generation
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Initialize generated sequence
        generated_ids = input_ids.clone()
        generated_text = prompt
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                
                # Get logits for the last token
                next_token_logits = outputs.logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated_ids[0].tolist()):
                        next_token_logits[token_id] /= repetition_penalty
                        
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k_logits, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    min_top_k = top_k_logits[-1]
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample and temperature > 0:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=-1)
                
                # Decode and yield new token
                new_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # Only yield the new part
                if len(new_text) > len(generated_text):
                    generated_text = new_text
                    yield generated_text


def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    checkpoint_path = Path("checkpoints/checkpoint-5000")
    
    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model config object
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    model_config = ModelConfig(config['model'])
    
    # Load model
    model = TransformerForCausalLM(model_config)
    
    # Load state dict from pytorch_model.bin
    model_state_dict = torch.load(
        checkpoint_path / "pytorch_model.bin", 
        map_location='cpu'
    )
    
    model.load_state_dict(model_state_dict, strict=False)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# Global variables for model and generator
model = None
tokenizer = None
generator = None

def initialize_model():
    """Initialize model and tokenizer."""
    global model, tokenizer, generator
    
    if model is None:
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = StreamingTextGenerator(model, tokenizer, device=device)
        print(f"Model loaded on {device}")


@spaces.GPU(duration=120) if HAS_SPACES else lambda x: x
def generate_response(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> Iterator[str]:
    """Generate streaming response."""
    
    # Initialize model if needed
    initialize_model()
    
    if not prompt.strip():
        yield "Please enter a prompt."
        return
    
    try:
        # Generate with streaming
        for partial_text in generator.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        ):
            yield partial_text
            
    except Exception as e:
        yield f"Error generating text: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Custom LLM - Text Generation", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 900px !important;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 Custom LLM - Foundational Language Model
            
            This is a custom-trained foundational language model with 2B parameters, 
            featuring modern transformer architecture with RoPE, RMSNorm, and SwiGLU.
            
            **Features:**
            - Streaming text generation
            - Configurable sampling parameters
            - ZeroGPU acceleration on Hugging Face Spaces
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Input section
                prompt_input = gr.Textbox(
                    lines=4,
                    placeholder="Enter your prompt here...",
                    label="Prompt",
                    show_copy_button=True,
                )
                
                # Generation parameters
                with gr.Accordion("Generation Parameters", open=False):
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=1024,
                            value=512,
                            step=1,
                            label="Max New Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="Temperature"
                        )
                    
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p (nucleus sampling)"
                        )
                        top_k = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=50,
                            step=5,
                            label="Top-k (0 = disabled)"
                        )
                    
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.05,
                        label="Repetition Penalty"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🚀 Generate", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=4):
                # Output section
                output_text = gr.Textbox(
                    lines=20,
                    label="Generated Text",
                    show_copy_button=True,
                    interactive=False
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Examples
        gr.Examples(
            examples=[
                ["Once upon a time in a distant galaxy,"],
                ["The future of artificial intelligence is"],
                ["In the year 2050, technology will"],
                ["Write a short story about a robot who"],
                ["Explain quantum computing in simple terms:"],
            ],
            inputs=[prompt_input],
            label="Example Prompts"
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_response,
            inputs=[
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            ],
            outputs=[output_text],
            show_progress=True,
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, output_text]
        )
        
        # Info section
        gr.Markdown(
            """
            ---
            
            **Model Details:**
            - Architecture: Custom Transformer with RoPE, RMSNorm, SwiGLU
            - Parameters: ~2B
            - Context Length: 2048 tokens
            - Tokenizer: GPT-2
            
            **Note:** This model runs on ZeroGPU when deployed on Hugging Face Spaces.
            Generation may take a few moments to start as the model loads on GPU.
            """
        )
    
    return demo


if __name__ == "__main__":
    # Initialize for local testing
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
    )