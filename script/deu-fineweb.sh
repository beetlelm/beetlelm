#!/usr/bin/env bash
set -euo pipefail

NPROC=8

# Toggle: also delete pretokenized caches after each run (expensive to regenerate).
# Set to 0 if you want to keep caches across runs.
CLEAN_PRETOK=1

cleanup_after_run() {
  local cfg="$1"
  echo "[cleanup] purging local state for ${cfg}"

  # Extract run_name and pretok cache dir from the YAML (simple grep, no yq needed)
  local run_name
  run_name=$(grep -E "^run_name:" "$cfg" | awk '{print $2}' | tr -d '"')

  if [[ -n "${run_name}" && -d "./runs/${run_name}" ]]; then
    echo "[cleanup] rm -rf ./runs/${run_name}"
    rm -rf "./runs/${run_name}"
  fi

  if [[ "${CLEAN_PRETOK}" == "1" ]]; then
    local pretok
    pretok=$(grep -E "^\s*pretokenized_cache_dir:" "$cfg" | awk '{print $2}' | tr -d '"')
    if [[ -n "${pretok}" && -d "${pretok}" ]]; then
      echo "[cleanup] rm -rf ${pretok}"
      rm -rf "${pretok}"
    fi
  fi
}

run_cfg() {
  local cfg="$1"
  torchrun --standalone --nproc_per_node=$NPROC script/train.py --config_path "$cfg"
  cleanup_after_run "$cfg"
}

echo "=============================="
echo "B1 (ENG -> DEU) - RUN FIRST"
echo "=============================="
run_cfg configs/fineweb_bilingual/core/B1/train_eng_L1-deu_L2.yaml

echo "=============================="
echo "B1 (DEU -> ENG)"
echo "=============================="
run_cfg configs/fineweb_bilingual/core/B1/train_deu_L1-eng_L2.yaml

for B in B2 B3 B4 B5; do
  echo "=============================="
  echo "DEU experiments: $B"
  echo "=============================="
  run_cfg "configs/fineweb_bilingual/core/${B}/train_deu_L1-eng_L2.yaml"
  run_cfg "configs/fineweb_bilingual/core/${B}/train_eng_L1-deu_L2.yaml"
done

echo "[done] all DEU runs complete; local runs/ and pretok_cache/ cleaned."
