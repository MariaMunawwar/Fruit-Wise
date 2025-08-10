#!/usr/bin/env bash
set -e
WEIGHTS_URL="https://github.com/MariaMunawwar/Fruit-Wise/releases/download/v1.0.0/model_final.pth"
DEST_DIR="$(cd "$(dirname "$0")"/.. && pwd)/models"
mkdir -p "$DEST_DIR"
echo "Downloading weights..."
curl -L "$WEIGHTS_URL" -o "$DEST_DIR/model_final.pth"
echo "Done: $DEST_DIR/model_final.pth"

