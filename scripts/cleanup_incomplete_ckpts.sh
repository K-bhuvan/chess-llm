#!/usr/bin/env bash
set -eu
SFT="${CHESS_OUTPUT:-$HOME/chess_outputs}/sft"
mkdir -p "$SFT"
for d in "$SFT"/checkpoint-*; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/adapter_model.safetensors" ]; then
    rm -rf "$d"
  fi
done
