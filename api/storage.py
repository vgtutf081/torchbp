from __future__ import annotations

from pathlib import Path


class ArtifactStorage:
    def store_file(self, *, job_id: str, local_path: Path, object_name: str) -> str:
        raise NotImplementedError


class LocalArtifactStorage(ArtifactStorage):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def store_file(self, *, job_id: str, local_path: Path, object_name: str) -> str:
        dest_dir = self.root_dir / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / object_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(local_path.read_bytes())
        return f"file://{dest_path.as_posix()}"


class S3ArtifactStorage(ArtifactStorage):
    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        endpoint_url: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def store_file(self, *, job_id: str, local_path: Path, object_name: str) -> str:
        key = f"jobs/{job_id}/{object_name}"
        self.client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"
