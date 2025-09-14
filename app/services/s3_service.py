import boto3
import os
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self, bucket_name: str, region: str = 'ap-northeast-2'):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    def upload_video(self, file_data: bytes, user_id: int, filename: str) -> Optional[str]:
        """
        Upload video to S3 and return the public URL

        Args:
            file_data: Video file bytes
            user_id: User ID for organizing files
            filename: Original filename

        Returns:
            S3 URL of the uploaded file or None if failed
        """
        try:
            # Generate unique S3 key with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_extension = os.path.splitext(filename)[1]
            s3_key = f"videos/user_{user_id}/{timestamp}_{filename}"

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_data,
                ContentType=self._get_content_type(file_extension),
                # Make it publicly readable (optional - remove if you want private files)
                ACL='public-read'
            )

            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Successfully uploaded video to S3: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload video to S3: {str(e)}")
            return None


    def delete_video(self, s3_url: str) -> bool:
        """
        Delete a video from S3

        Args:
            s3_url: Full S3 URL of the video

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key from URL
            s3_key = s3_url.split('.com/')[-1]

            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            logger.info(f"Successfully deleted video from S3: {s3_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete video from S3: {str(e)}")
            return False

    def _get_content_type(self, file_extension: str) -> str:
        """Get appropriate content type for video files"""
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska'
        }
        return content_types.get(file_extension.lower(), 'video/mp4')

# Singleton instance
s3_service = None

def get_s3_service():
    global s3_service
    if s3_service is None:
        bucket_name = os.getenv('S3_BUCKET_NAME', 'hope-api-videos')
        region = os.getenv('AWS_REGION', 'ap-northeast-2')
        s3_service = S3Service(bucket_name, region)
    return s3_service