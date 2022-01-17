# Base image
FROM python:3.9-slim


# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt install -y wget && \
apt clean && rm -rf /var/lib/apt/lists/*


COPY run_docker.sh run_docker.sh
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

# should NOT be done during deployement
COPY velvety-calling-337909-e89e3748a258.json velvety-calling-337909-e89e3748a258.json
ENV GOOGLE_APPLICATION_CREDENTIALS=velvety-calling-337909-e89e3748a258.json

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir  # --no-cache-dir: don't store installation/source files --> keep docker image small!
RUN pip install --upgrade google-cloud-storage --no-cache-dir

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup


# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# -u ensure that output of script is redirected to console
ENTRYPOINT ["bash", "run_docker.sh"]
