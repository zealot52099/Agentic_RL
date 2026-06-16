#!/usr/bin/env bash
set -euo pipefail

SOURCE=/publicdata/huggingface.co/Qwen/Qwen3-1.7B-Base
TARGET=/tmp/agentic_rl_models/Qwen3-1.7B-Base

mkdir -p "$TARGET"

for file in \
  config.json \
  generation_config.json \
  merges.txt \
  model.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json
do
  cp -f "$SOURCE/$file" "$TARGET/$file"
done

source_size=$(stat -c %s "$SOURCE/model.safetensors")
target_size=$(stat -c %s "$TARGET/model.safetensors")
if [[ "$source_size" != "$target_size" ]]; then
  echo "model.safetensors size mismatch: source=$source_size target=$target_size" >&2
  exit 1
fi

sha256sum "$SOURCE/model.safetensors" "$TARGET/model.safetensors"
ls -lah "$TARGET"
