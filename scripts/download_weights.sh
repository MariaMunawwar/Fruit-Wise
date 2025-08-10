#!/usr/bin/env bash
set -e
WEIGHTS_URL="<<PUT_PUBLIC_URL_TO_model_final.pth_HERE>>"
DEST_DIR="$(cd "$(dirname "$0")"/.. && pwd)/models"
mkdir -p "$DEST_DIR"
echo "Downloading weights..."
curl -L "$WEIGHTS_URL" -o "$DEST_DIR/model_final.pth"
echo "Done: $DEST_DIR/model_final.pth"

