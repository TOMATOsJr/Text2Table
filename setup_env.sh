#!/bin/bash

# Configuration for INLP Project Scratch Storage
SCRATCH_DIR="/scratch/INLP_Project_Wiki"
CACHE_DIR="$SCRATCH_DIR/cache"

# Create directories if they don't exist
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/torch"
mkdir -p "$CACHE_DIR/sentence_transformers"

# Redirect HuggingFace cache
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"

# Redirect PyTorch and SentenceTransformers cache
export TORCH_HOME="$CACHE_DIR/torch"
export SENTENCE_TRANSFORMERS_HOME="$CACHE_DIR/sentence_transformers"

echo "✅ Environment redirected to $SCRATCH_DIR"
echo "   HF_HOME: $HF_HOME"
echo "   TORCH_HOME: $TORCH_HOME"
