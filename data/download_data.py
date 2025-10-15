import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm
from dotenv import load_dotenv

# --- Cargar variables de entorno desde archivo .env ---
load_dotenv(override=True)

# --- Configuración del bucket y credenciales ---
# Try to get credentials from environment variables first, fallback to hardcoded
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = "anyoneai-datasets"
PREFIX = "SKU-110K/SKU110K_fixed/"
DEST_DIR = "data/SKU110K_dataset"

def download_dataset():
    """
    Download the SKU-110K dataset from AWS S3 bucket
    """
    try:
        # Validate credentials
        if not ACCESS_KEY or not SECRET_KEY:
            print("Error: AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            return
        
        print(f"Using AWS credentials: {ACCESS_KEY[:10]}...")
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )
        
        # Test credentials by trying to list bucket
        print("Testing AWS credentials...")
        s3_client.head_bucket(Bucket=BUCKET)
        print("✓ AWS credentials are valid")
        
        # Create destination directory if it doesn't exist
        os.makedirs(DEST_DIR, exist_ok=True)
        
        # List all objects in the bucket with the specified prefix
        print(f"Listing objects in bucket '{BUCKET}' with prefix '{PREFIX}'...")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
        
        # Get all object keys
        object_keys = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    object_keys.append(obj['Key'])
        
        if not object_keys:
            print(f"No objects found with prefix '{PREFIX}'")
            return
        
        print(f"Found {len(object_keys)} objects to download")
        
        # Download each object
        for key in tqdm(object_keys, desc="Downloading files"):
            # Create local file path
            local_file_path = os.path.join(DEST_DIR, key.replace(PREFIX, ""))
            
            # Create directory structure if needed
            local_dir = os.path.dirname(local_file_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            # Download the file
            try:
                s3_client.download_file(BUCKET, key, local_file_path)
            except ClientError as e:
                print(f"Error downloading {key}: {e}")
                continue
        
        print(f"Download completed! Files saved to: {DEST_DIR}")
        
    except NoCredentialsError:
        print("Error: AWS credentials not found or invalid")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_dataset()
