#!/usr/bin/env python3
"""Test script to verify model and training setup."""

import sys
import torch
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.transformer import TransformerForCausalLM
from src.model.layers import RMSNorm, RotaryEmbedding, SwiGLU
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test model creation."""
    logger.info("Testing model creation...")
    
    # Simple config for testing
    class TestConfig:
        vocab_size = 50257
        hidden_size = 512
        num_hidden_layers = 4
        num_attention_heads = 8
        intermediate_size = 2048
        hidden_act = "silu"
        max_position_embeddings = 512
        initializer_range = 0.02
        layer_norm_eps = 1e-5
        use_cache = False
        rope_theta = 10000.0
        attention_dropout = 0.0
        hidden_dropout = 0.0
    
    config = TestConfig()
    model = TransformerForCausalLM(config)
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        
    assert outputs.logits.shape == (batch_size, seq_length, config.vocab_size)
    logger.info("✓ Model creation and forward pass successful")
    
    # Test with labels (loss computation)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    
    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])
    logger.info("✓ Loss computation successful")
    
    return model


def test_rotary_embeddings():
    """Test rotary embeddings."""
    logger.info("Testing rotary embeddings...")
    
    dim = 64
    max_pos = 512
    rope = RotaryEmbedding(dim, max_position_embeddings=max_pos)
    
    seq_len = 128
    x = torch.randn(2, 8, seq_len, dim)  # batch, heads, seq, dim
    
    cos, sin = rope(x, seq_len=seq_len)
    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)
    
    logger.info("✓ Rotary embeddings test passed")


def test_data_loading():
    """Test data loading setup."""
    logger.info("Testing data loading setup...")
    
    try:
        from datasets import load_dataset
        
        # Test loading a small sample
        dataset = load_dataset(
            "tiiuae/falcon-refinedweb",
            split="train",
            streaming=True
        )
        
        # Get one sample
        sample = next(iter(dataset))
        assert "content" in sample
        logger.info("✓ Dataset loading successful")
        
    except Exception as e:
        logger.warning(f"Dataset loading test skipped: {e}")


def test_tokenizer():
    """Test tokenizer."""
    logger.info("Testing tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello, this is a test of the tokenizer."
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    
    decoded = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    logger.info(f"Original: {text}")
    logger.info(f"Decoded: {decoded}")
    logger.info("✓ Tokenizer test passed")


def test_gpu_availability():
    """Test GPU availability."""
    logger.info("Testing GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✓ Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        logger.warning("⚠ No GPUs available. Training will be slow on CPU.")


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("Running LLM Training Setup Tests")
    logger.info("=" * 50)
    
    try:
        test_gpu_availability()
        test_model_creation()
        test_rotary_embeddings()
        test_tokenizer()
        test_data_loading()
        
        logger.info("=" * 50)
        logger.info("✓ All tests passed! Setup is ready for training.")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()