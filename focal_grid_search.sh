#!/bin/bash

# Hyperparameter grid
ALPHAS=(0.1 0.25 0.5)
GAMMAS=(1.0 2.0 3.0)
ALPHA_ADVS=(1.0 2.0 5.0)
OVERSAMPLE_RATES=(1 3 5)
UNDERSAMPLE_RATES=(0.25 0.5 1.0)

# Fixed params
MODEL="compgcn"
TARGET="83332"

# Loop over combinations
for ALPHA in "${ALPHAS[@]}"; do
  for GAMMA in "${GAMMAS[@]}"; do
    for ADV in "${ALPHA_ADVS[@]}"; do
      for OVER in "${OVERSAMPLE_RATES[@]}"; do
        for UNDER in "${UNDERSAMPLE_RATES[@]}"; do
          NAME="a${ALPHA}_g${GAMMA}_adv${ADV}_over${OVER}_under${UNDER}"
          LOGFILE="logs/${NAME}.log"
          mkdir -p logs
          echo "[+] Running config: $NAME"

          python train_and_eval.py \
            -m $MODEL \
            -r 1 \
            --target $TARGET \
            --oversample_rate $OVER \
            --undersample_rate $UNDER \
            --quiet \
            --alpha $ALPHA \
            --gamma $GAMMA \
            --alpha_adv $ADV \
            &> "$LOGFILE"
        done
      done
    done
  done
done