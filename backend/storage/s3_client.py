import os
import time

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Crée un client boto3 compatible MinIO / S3."""

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def wait_for_minio(max_retries: int = 10, delay: int = 2):
    """Attend que MinIO soit disponible avant de continuer."""

    client = get_s3_client()

    for attempt in range(1, max_retries + 1):
        try:
            client.list_buckets()
            print("MinIO disponible.")
            return client
        except Exception as error:
            print(f"MinIO indisponible ({attempt}/{max_retries}) : {error}")
            time.sleep(delay)

    raise RuntimeError("MinIO inaccessible après plusieurs tentatives.")


def download_model(local_path: str) -> str:
    """Télécharge le modèle depuis MinIO vers un chemin local."""

    bucket = os.getenv("MINIO_BUCKET_MODELS")
    object_name = os.getenv("MODEL_OBJECT_NAME")

    client = wait_for_minio()

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        client.download_file(bucket, object_name, local_path)
        print(f"Modèle téléchargé : s3://{bucket}/{object_name} → {local_path}")
        return local_path

    except ClientError as error:
        raise FileNotFoundError(
            f"Modèle s3://{bucket}/{object_name} introuvable dans MinIO. "
            "Vérifie que scripts/upload_model_to_minio.py a bien été exécuté."
        ) from error
