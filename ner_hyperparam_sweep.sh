cuda="$1"
#checkpoint="$2"

if [[ $# -ne 1 ]] ; then
  echo 'Usage: sweep.sh <cuda'
  exit 1
fi

CUDA_VISIBLE_DEVICES="$cuda" python scripts/run_ner_training.py cuda="$cuda" learning_rate=5e-6,1e-5,5e-5,1e-4,5e-4 label_smoothing_factor=0.0,0.05,0.1,0.2 weight_decay=0.0,0.05,0.1,0.2,0.3,0.5,1.0 lr_scheduler_type=cosine_with_restarts,linear,constant_with_warmup warmup_ratio=0.0,0.05,0.1 batch_size=16,32 -m
