#!bin/bash
echo "experiments: $1"
echo "wandb_api_key: $2"
echo "Start downloading data..."
gsutil cp gs://mnist-new-bucket/train data/processed/train

echo "Start training script..."
python -u src/models/train_model.py
