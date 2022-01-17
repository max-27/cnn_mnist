#!bin/bash

export JOB_NAME=training_go
export REGION=us-central1
export JOB_DIR="gs://mnist-new-bucket/models"

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  experiment=exp1 \
  save_to_gs=True \
  bucket_name="mnist-new-bucket"

