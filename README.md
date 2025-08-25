# VelocityLM

A high-performance, custom transformer language model trained from scratch using modern architectural innovations. VelocityLM combines state-of-the-art techniques including RMSNorm, SwiGLU activation, and Rotary Position Embeddings (RoPE) to deliver efficient and scalable language modeling.

## 🚀 Try the Model

- **Interactive Demo**: [HuggingFace Space](https://huggingface.co/spaces/dixisouls/VelocityLM)
- **Model Repository**: [HuggingFace Model](https://huggingface.co/dixisouls/VelocityLM)

## 🏗️ Architecture

VelocityLM features a custom transformer architecture optimized for performance and efficiency:

### Model Specifications
- **Parameters**: ~2B parameters (2048 hidden size, 24 layers)
- **Architecture**: Decoder-only transformer with causal attention
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)
- **Context Length**: 2,048 tokens
- **Attention Heads**: 32 heads per layer
- **Intermediate Size**: 8,192 (4x hidden size)

### Key Innovations

#### 1. RMSNorm (Root Mean Square Normalization)
- Replaces LayerNorm for improved training stability
- More efficient computation with better gradient flow

#### 2. SwiGLU Activation Function
- Gated Linear Unit with Swish activation
- Superior to standard ReLU/GELU for language modeling
- Provides better expressivity and gradient flow

#### 3. Rotary Position Embeddings (RoPE)
- Relative position encoding that preserves rotational invariance
- Better extrapolation to longer sequences
- More efficient than learned absolute positions

#### 4. Gradient Checkpointing
- Memory-efficient training for large models
- Trades computation for memory during backpropagation
- Essential for fitting large models in GPU memory

## 📁 Project Structure

```
custom_llm/
├── configs/
│   └── train_config.yaml          # Training configuration
├── src/
│   ├── model/
│   │   ├── transformer.py         # Main transformer architecture
│   │   └── layers.py             # Custom layers (RMSNorm, RoPE, SwiGLU)
│   ├── data/
│   │   └── dataset.py            # Streaming dataset implementation
│   ├── training/
│   │   └── trainer.py            # Distributed training logic
│   └── inference/
│       └── inference.py          # Text generation utilities
├── scripts/
│   ├── train.py                  # Main training script
│   └── test_model.py            # Model testing utilities
├── checkpoints/                  # Model checkpoints
│   └── checkpoint-5000/
└── requirements.txt             # Project dependencies
```

## 🔧 Technical Details

### Training Infrastructure
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Mixed Precision**: FP16 training with automatic loss scaling
- **Gradient Accumulation**: Configurable steps for effective batch scaling
- **Checkpointing**: Automatic model and optimizer state saving

### Data Pipeline
- **Dataset**: Trained on [Falcon RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) (high-quality web text)
- **Streaming**: Memory-efficient streaming data loader
- **Tokenization**: GPT-2 tokenizer with padding and attention masks
- **Processing**: On-the-fly tokenization with configurable sequence lengths

### Optimization
- **Optimizer**: AdamW with separate weight decay for parameters
- **Scheduler**: Cosine annealing with warm restarts
- **Learning Rate**: 6e-4 with warmup (2000 steps) and decay to 6e-5
- **Gradient Clipping**: Max norm of 1.0 for training stability

### Memory Optimizations
- **Gradient Checkpointing**: Reduces memory usage by ~40%
- **Parameter Grouping**: Separate weight decay for different parameter types
- **Efficient Attention**: Optimized causal self-attention implementation
- **Streaming Data**: No dataset preloading, processes on-the-fly

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- Multiple GPUs recommended for training

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd VelocityLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training

#### Single GPU Training
```bash
python scripts/train.py --config configs/train_config.yaml --num-gpus 1
```

#### Multi-GPU Distributed Training
```bash
python scripts/train.py --config configs/train_config.yaml --num-gpus 4
```

#### Custom Training Parameters
```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --max-samples 50000 \
  --num-gpus 2
```

### Inference

#### Interactive Generation
```bash
python src/inference/inference.py \
  --checkpoint checkpoints/checkpoint-5000 \
  --prompt "The future of artificial intelligence" \
  --max-length 200 \
  --temperature 0.8
```

#### Programmatic Usage
```python
from src.inference.inference import load_generator

# Load the model
generator = load_generator('checkpoints/checkpoint-5000')

# Generate text
response = generator.generate(
    "Once upon a time",
    max_length=150,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
print(response[0])
```

### Testing Setup
```bash
python scripts/test_model.py
```

## ⚙️ Configuration

The training configuration is managed through `configs/train_config.yaml`:

```yaml
model:
  hidden_size: 2048
  num_hidden_layers: 24
  num_attention_heads: 32
  intermediate_size: 8192
  max_position_embeddings: 2048

training:
  batch_size_per_device: 4
  gradient_accumulation_steps: 8
  learning_rate: 0.0006
  num_training_steps: 100000
  fp16: true
  gradient_checkpointing: true

data:
  dataset_name: "tiiuae/falcon-refinedweb"  # Falcon RefinedWeb dataset
  max_seq_length: 2048
```

## 🔍 Model Performance

### Training Metrics
- **Training Steps**: 5,000+ completed
- **Dataset**: [Falcon RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) (high-quality web text)
- **Batch Size**: 32 effective (4 per device × 8 accumulation × 1 GPU)

### Training Hardware
- **CPU**: 2 x AMD EPYC 9334
- **GPU**: 4 x NVIDIA A100 (80GB)
- **Training Infrastructure**: High-performance GPU cluster optimized for large-scale model training

## 🧪 Advanced Features

### Generation Strategies
- **Greedy Decoding**: Deterministic, highest probability tokens
- **Top-k Sampling**: Sample from top-k most likely tokens
- **Top-p (Nucleus) Sampling**: Dynamic vocabulary based on cumulative probability
- **Temperature Control**: Adjust randomness of generation
- **Repetition Penalty**: Reduce repetitive text generation

### Distributed Training Features
- **Multi-Node Support**: Scale across multiple machines
- **Gradient Synchronization**: Efficient parameter updates
- **Automatic Mixed Precision**: Memory and speed optimization
- **Fault Tolerance**: Checkpoint-based recovery

## 📊 Monitoring and Logging

### TensorBoard Integration
```bash
tensorboard --logdir logs/
```

### Weights & Biases (Optional)
Configure W&B API key for advanced experiment tracking:
```bash
export WANDB_API_KEY=your_key_here
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📈 Roadmap

- [ ] Flash Attention 2.0 integration
- [ ] Support for longer context lengths (4K+)
- [ ] Model quantization for deployment
- [ ] Fine-tuning scripts for downstream tasks
- [ ] ONNX export for production inference
- [ ] Multi-modal capabilities

## 🐛 Known Issues

- Checkpoint loading requires exact architecture match
- Streaming dataset may be slow on first epoch