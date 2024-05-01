#!/bin/bash

MODEL_DIR=model
REPO_PATH="https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main"
CLIP_MODEL=mmproj-model-f16.gguf
LLAVA_MODEL=ggml-model-q4_k.gguf

mkdir -p $MODEL_DIR

if [ ! -f "$MODEL_DIR/$CLIP_MODEL" ]; then
    wget $REPO_PATH/$CLIP_MODEL?download=true -O $CLIP_MODEL
    mv $CLIP_MODEL $MODEL_DIR/
fi

if [ ! -f "$MODEL_DIR/$LLAVA_MODEL" ]; then
    wget $REPO_PATH/$LLAVA_MODEL?download=true -O $LLAVA_MODEL
    mv $LLAVA_MODEL $MODEL_DIR/
fi
