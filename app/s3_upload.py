#!/usr/bin/env python3
import os
import re
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def parse_basename(folder):
    """
    Scans the folder for a file matching the pattern 'HRC_YYYYMMDDT[HHMM]'
    and returns the base name.
    """
    pattern = re.compile(r"^(HRC_\d{8}T\d{4})")
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            return match.group(1)
    return None

def upload_files_to_s3(folder, bucket, base_name, region):
    """
    Uploads all files in the folder that start with the base_name to the specified S3 bucket,
    placing them under a folder (key prefix) named after the base_name.
    """
    s3_key = os.environ.get("S3_MEDIA_KEY")
    s3_secret = os.environ.get("S3_MEDIA_SECRET")

    s3_client = boto3.client('s3', aws_access_key_id=s3_key, aws_secret_access_key=s3_secret, region_name=region)


    # Gather all files that start with the base name
    files_to_upload = [f for f in os.listdir(folder) if f.startswith(base_name)]
    if not files_to_upload:
        print(f"No files starting with '{base_name}' found in {folder}")
        return
    
    for file in files_to_upload:
        file_path = os.path.join(folder, file)
        s3_key = f"{base_name}/{file}"  # Create a folder in S3 named after the base name
        try:
            s3_client.upload_file(file_path, bucket, s3_key)
            print(f"Uploaded '{file}' to s3://{bucket}/{s3_key}")
        except (NoCredentialsError, ClientError) as e:
            print(f"Failed to upload '{file}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Upload session files to an S3 bucket under a folder named by the base name."
    )
    parser.add_argument("folder", help="Path to the folder containing the session files")
    args = parser.parse_args()
    
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        return
    
    # Retrieve environment variables for the bucket and datacenter (region)
    bucket = os.environ.get("S3_MEDIA_BUCKET")
    if not bucket:
        print("Error: BUCKET_LINK environment variable not set.")
        return
    
    region = os.environ.get("S3_MEDIA_REGION", "us-east-1")
    
    base_name = parse_basename(folder)
    if not base_name:
        print("Error: Could not find a file matching the expected pattern 'HRC_YYYYMMDDT[HHMM]' in the folder.")
        return
    
    upload_files_to_s3(folder, bucket, base_name, region)

if __name__ == "__main__":
    main()

