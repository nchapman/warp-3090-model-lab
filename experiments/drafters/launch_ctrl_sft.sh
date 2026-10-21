#!/bin/bash
# launch_ctrl.sh — static-control SFT arm: champion recipe verbatim
# (fixed-exp-decay gamma 7, default loss-fn) on the regen-v4 corpus, 3 epochs.
# Discriminates "D-PACE+CE fails" vs "corpus fails". Writes PID to /root/ctrl.pid.
set -e
cd /root/warp-3090-quant-lab
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
VPATH=$(cat /root/verifier_path)
echo "speculators train --speculator-type dflash2 --draft-attn-impl simple_flex_attention --from-pretrained draft_init --verifier-name-or-path $VPATH --data-path ds_regen4b_sft --epochs 3 --lr 6e-4 --scheduler-type cosine --scheduler-warmup-ratio 0.04 --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 7 --max-anchors 512 --checkpoint-freq 1 --train-data-ratio 0.97 --log-freq 10 --save-path draft_ctrl4b" > draft_ctrl4b_train_command.txt
nohup .venv-dflash/bin/speculators train \
  --speculator-type dflash2 \
  --draft-attn-impl simple_flex_attention \
  --from-pretrained draft_init \
  --verifier-name-or-path "$VPATH" \
  --data-path ds_regen4b_sft \
  --epochs 3 --lr 6e-4 --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
  --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 7 \
  --max-anchors 512 --checkpoint-freq 1 \
  --train-data-ratio 0.97 --log-freq 10 \
  --save-path draft_ctrl4b >> /root/chain_ctrl.log 2>&1 &
echo $! > /root/ctrl.pid
echo "ctrl trainer pid $(cat /root/ctrl.pid)"
