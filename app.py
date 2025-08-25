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
    
    # Custom CSS for enhanced UI
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    .header-text {
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em !important;
        font-weight: bold !important;
        margin-bottom: 0.5em !important;
    }
    
    .subtitle-text {
        text-align: center;
        color: #666;
        font-size: 1.2em !important;
        margin-bottom: 2em !important;
    }
    
    .parameter-box {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 1px solid #4a5568 !important;
    }

    .parameter-box summary {
        color: #ffffff !important;
        font-weight: bold !important;
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 10px !important;
        border-radius: 10px !important;
    }

    .parameter-box details summary {
        color: #ffffff !important;
        font-weight: bold !important;
    }

    /* Make ALL text white in the parameter box */
    .parameter-box,
    .parameter-box *,
    .parameter-box label,
    .parameter-box span,
    .parameter-box p,
    .parameter-box div,
    .parameter-box small {
        color: #ffffff !important;
    }

    /* Ensure input values are also white */
    .parameter-box input[type="number"],
    .parameter-box .gr-textbox input {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid #4a5568 !important;
    }

    /* Make the centered description text white too */
    .parameter-box > p {
        color: #ffffff !important;
        text-align: center !important;
    }
    
    .output-box {
        border-radius: 15px !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    .generate-btn {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        padding: 15px 30px !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .clear-btn {
        background: linear-gradient(45deg, #ff6b6b 0%, #ee5a24 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        padding: 10px 20px !important;
        box-shadow: 0 2px 10px rgba(255, 107, 107, 0.3) !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 1px solid #f0c27b !important;
        margin-top: 20px !important;
    }
    
    .example-box {
        background: linear-gradient(135def, #e8f5e8 0%, #d4edda 100%) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        border: 1px solid #c3e6cb !important;
    }
    
    .metric-card {
        background: white !important;
        border-radius: 10px !important;
        padding: 15px !important;
        text-align: center !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        border-left: 4px solid #667eea !important;
    }
    
    .progress-bar {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
    }
    """
    
    with gr.Blocks(
        title="🤖 Custom LLM - Advanced Text Generation", 
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray"
        ),
        css=custom_css
    ) as demo:
        
        # Header with gradient text
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="header-text">🤖 Custom LLM</h1>
            <p class="subtitle-text">Advanced 2B Parameter Foundational Language Model</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 1.5rem 0;">
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">2B+</h3>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Parameters</p>
                </div>
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">2048</h3>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Context Length</p>
                </div>
            </div>
        </div>
        """)
        
        gr.Markdown(
            """
            <div style="text-align: center; background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%); 
                        padding: 20px; border-radius: 15px; margin-bottom: 2rem; border: 1px solid #e1e8f7;">
                <p style="margin: 0; font-size: 1.1em; color: #4a5568;">
                    🎯 <strong>Modern Architecture:</strong> RoPE • RMSNorm • SwiGLU • Multi-Head Attention<br>
                    ✨ <strong>Features:</strong> Text Generation • Configurable Sampling • GPU Accelerated
                </p>
            </div>
            """, 
            elem_classes=["info-box"]
        )
        
        with gr.Row(equal_height=True):
            # Input Column
            with gr.Column(scale=2, min_width=400):
                gr.HTML("<div style='margin-bottom: 1rem;'><h3 style='color: #667eea; margin: 0;'>💬 Input Prompt</h3></div>")
                
                prompt_input = gr.Textbox(
                    lines=6,
                    placeholder="✨ Enter your creative prompt here...\n\nExample: Write a story about a future where AI and humans collaborate to solve climate change...",
                    label="Your Prompt",
                    show_copy_button=True,
                    container=True,
                    elem_classes=["input-box"]
                )
                
                # Advanced Parameters Section
                with gr.Accordion("🎛️ Advanced Generation Parameters", open=False, elem_classes=["parameter-box"]):
                    gr.HTML("<p style='text-align: center; color: #333; margin-bottom: 1rem;'>Fine-tune your generation settings</p>")
                    
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=1024,
                            value=512,
                            step=1,
                            label="🔢 Max New Tokens",
                            info="Maximum number of tokens to generate"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="🌡️ Temperature",
                            info="Higher = more creative, lower = more focused"
                        )
                    
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="🎯 Top-p",
                            info="Nucleus sampling threshold"
                        )
                        top_k = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=50,
                            step=5,
                            label="📊 Top-k",
                            info="Top-k sampling limit (0 = disabled)"
                        )
                    
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.05,
                        label="🔄 Repetition Penalty",
                        info="Reduce repetitive text (higher = less repetition)"
                    )
                
                # Generate Button with enhanced styling
                gr.HTML("<div style='margin: 1.5rem 0;'>")
                generate_btn = gr.Button(
                    "🚀 Generate Text",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                    scale=1
                )
                gr.HTML("</div>")
                
                # Quick Settings Presets
                gr.HTML("<div style='margin-top: 1rem;'><h4 style='color: #667eea; margin-bottom: 0.5rem;'>⚡ Quick Presets</h4></div>")
                with gr.Row():
                    creative_btn = gr.Button("🎨 Creative", size="sm", variant="secondary")
                    balanced_btn = gr.Button("⚖️ Balanced", size="sm", variant="secondary") 
                    precise_btn = gr.Button("🎯 Precise", size="sm", variant="secondary")
                
            # Output Column
            with gr.Column(scale=3, min_width=500):
                gr.HTML("<div style='margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;'><h3 style='color: #667eea; margin: 0;'>📝 Generated Output</h3></div>")
                
                output_text = gr.Textbox(
                    lines=22,
                    label="Generated Text",
                    show_copy_button=True,
                    interactive=False,
                    placeholder="Your generated text will appear here...\n\n✨ Streaming in real-time\n🚀 Powered by custom 2B parameter model",
                    elem_classes=["output-box"],
                    container=True
                )
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", elem_classes=["clear-btn"])
        
        # Enhanced Examples Section
        gr.HTML("<div style='margin: 2rem 0;'><h3 style='color: #667eea; text-align: center; margin-bottom: 1rem;'>🎯 Example Prompts</h3></div>")
        
        with gr.Accordion("📚 Prompt Examples", open=True, elem_classes=["example-box"]):
            gr.Examples(
                examples=[
                    ["Once upon a time in a distant galaxy, there lived a civilization that had never seen the stars."],
                    ["The old lighthouse keeper noticed something strange about the fog that night."],
                    ["In the depths of the Amazon rainforest, Dr. Martinez made a discovery that would change everything."],
                    ["The last bookstore on Earth was about to close its doors forever when"],
                    ["As the spaceship approached the mysterious planet, the crew realized"],
                    ["The clockmaker's shop had been abandoned for fifty years, but every morning at precisely 9 AM"],
                    ["Deep beneath the city, in tunnels forgotten by time, archaeologist Elena found"],
                    ["The message in a bottle had traveled across three oceans before washing ashore"],
                ],
                inputs=[prompt_input],
                label="Click any example to get started!",
                examples_per_page=4
            )
        
        # Event handlers for main functionality
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
        
        # Preset button handlers
        creative_btn.click(
            fn=lambda: (1.2, 0.95, 40, 1.05),
            outputs=[temperature, top_p, top_k, repetition_penalty]
        )
        
        balanced_btn.click(
            fn=lambda: (0.8, 0.9, 50, 1.1),
            outputs=[temperature, top_p, top_k, repetition_penalty]
        )
        
        precise_btn.click(
            fn=lambda: (0.3, 0.8, 20, 1.2),
            outputs=[temperature, top_p, top_k, repetition_penalty]
        )
        
        # Utility button handlers
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, output_text]
        )
        
    
    return demo


if __name__ == "__main__":
    # Initialize for local testing
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
    )