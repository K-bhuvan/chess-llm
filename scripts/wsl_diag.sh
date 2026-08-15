#!/usr/bin/env bash
set -u
echo "=== uptime ==="
uptime
echo "=== free ==="
free -h
echo "=== oom / kill ==="
dmesg -T 2>/dev/null | grep -iE 'oom|killed process|out of memory|chess_llm|python' | tail -n 30
echo "=== last python ==="
ls -l /home/bhuvan/chess_outputs/train_full.log 2>/dev/null
