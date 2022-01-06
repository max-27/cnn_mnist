# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir  # --no-cache-dir: don't store installation/source files --> keep docker image small!

ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]


## run like this
# docker run --rm \
# -v $(pwd)/models/model_e5_b64_lr0.001_5.pth:/models/model_e5_b64_lr0.001_5.pth \
# -v $(pwd)/data/processed/test:/test \
# predict:latest \
# --model-path ../models/model_e5_b64_lr0.001_5.pth \
# --data-path data/processed/test

# mounting documentation: https://docs.docker.com/storage/volumes/

