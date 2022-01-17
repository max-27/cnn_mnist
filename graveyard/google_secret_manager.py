# Import the Secret Manager client library.
from google.cloud import secretmanager

# GCP project in which to store secrets in Secret Manager.
project_id = "velvety-calling-337909"

# ID of the secret to create.
secret_id = "wandb_api"

# Create the Secret Manager client.
client = secretmanager.SecretManagerServiceClient()

name = f"projects/{project_id}/secrets/wandb_api/versions/latest"

# Access the secret version.
response = client.access_secret_version(name=name)

print(response.payload.data.decode("utf-8"))
