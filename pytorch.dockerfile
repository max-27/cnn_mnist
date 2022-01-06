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
RUN pip install -r notorch_requirements.txt --no-cache-dir  # --no-cache-dir: don't store installation/source files --> keep docker image small!

# -u ensure that output of script is redirected to console
ENTRYPOINT ["python", "-u", "test_image.py"]