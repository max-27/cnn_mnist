# Import the Secret Manager client library.
from google.cloud import secretmanager, storage


def get_wandb_api_key():
    # GCP project in which to store secrets in Secret Manager.
    project_id = "velvety-calling-337909"
    # ID of the secret to create.
    secret_id = "wandb_api"
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/wandb_api/versions/latest"
    # Access the secret version.
    return client.access_secret_version(name=name).payload.data.decode("utf-8")


def upload_blob(bucket_name, model_filename):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_filename)


def upload_to_gs(file, bucket_name, project_name):
    client = storage.Client(project=project_name)
    bucket = client.get_bucket(bucket_name)
    encryption_key = "fl4291954r5adre2c8981gb9t7e73b99"
    blob = storage.Blob("secure-data", bucket, encryption_key=encryption_key)
    blob.upload_from_file(file)
