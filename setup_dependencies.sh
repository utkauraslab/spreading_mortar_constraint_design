#!/bin/bash

# This script sets up the necessary dependencies for the project.
# It initializes the git submodules and downloads the required model weights.

# --- Color Codes for Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting dependency setup...${NC}"

# --- 1. Initialize Git Submodules ---
# Ensures that the submodule code is downloaded.
echo -e "\n${YELLOW}Updating Git submodules...${NC}"
git submodule update --init --recursive
echo -e "${GREEN}Submodules are up to date.${NC}"


# --- 2. Create Checkpoint Directories ---
# Define paths based on the submodule folder names
COTRACKER_CKPT_DIR="co-tracker/checkpoints"
DEPTH_ANYTHING_CKPT_DIR="Depth-Anything-V2/checkpoints"

echo -e "\n${YELLOW}Creating checkpoint directories...${NC}"
mkdir -p "$COTRACKER_CKPT_DIR"
mkdir -p "$DEPTH_ANYTHING_CKPT_DIR"
echo "Created directories if they didn't exist."


# --- 3. Download Model Weights ---
echo -e "\n${YELLOW}Downloading model weights...${NC}"

# --- CoTracker Model ---
COTRACKER_URL="https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COTRACKER_FILE="$COTRACKER_CKPT_DIR/scaled_offline.pth"

if [ -f "$COTRACKER_FILE" ]; then
    echo "CoTracker (scaled_offline) model already exists. Skipping download."
else
    echo "Downloading CoTracker (scaled_offline) model..."
    wget -O "$COTRACKER_FILE" "$COTRACKER_URL"
    echo "CoTracker (scaled_offline) model downloaded."
fi


# --- Depth Anything V2 Model (ViT-Large) ---
# URL for the Large model
DEPTH_ANYTHING_URL="https://huggingface.co/LiheYoung/depth-anything-v2-storage/resolve/main/depth_anything_v2_vitl.pth"
# Filename for the Large model
DEPTH_ANYTHING_FILE="$DEPTH_ANYTHING_CKPT_DIR/depth_anything_v2_vitl.pth"

if [ -f "$DEPTH_ANYTHING_FILE" ]; then
    echo "Depth Anything V2 (Large) model already exists. Skipping download."
else
    echo "Downloading Depth Anything V2 (Large) model..."
    wget -O "$DEPTH_ANYTHING_FILE" "$DEPTH_ANYTHING_URL"
    echo "Depth Anything V2 (Large) model downloaded."
fi


echo -e "\n${GREEN}Setup complete! All dependencies are installed.${NC}"