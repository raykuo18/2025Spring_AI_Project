#!/bin/bash

# Base directory relative to current directory
BASE_DIR="./models"
mkdir -p "$BASE_DIR"

echo "Downloading LLaMA 2 65B q4_k_m model..."
cd "$BASE_DIR"
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-65B-GPTQ llama-2-65b-gptq

echo "Downloading Mixtral 8x7B Instruct model..."
git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral-8x7b-instruct

echo "✅ Models downloaded to $BASE_DIR"
