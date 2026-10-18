"""
GraphRAG Backend - Cloud File Storage Service (Zoho Catalyst File Store)
Manages document storage across local filesystem and Zoho Catalyst Cloud File Store.
Supports automatic OAuth2 token refresh, file upload, file retrieval, and deletion.
"""

import os
import time
import mimetypes
from typing import Optional, Dict, Any, Tuple
import aiofiles
import httpx
from loguru import logger

from app.config import settings


class StorageService:
    """
    Unified Storage Service supporting:
    - Zoho Catalyst Cloud File Store (when configured via .env)
    - Local disk fallback (when Catalyst credentials are not yet populated)
    """

    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # -------------------------------------------------------------------------
    # Configuration & Health Checks
    # -------------------------------------------------------------------------

    def is_catalyst_configured(self) -> bool:
        """Check if all required Zoho Catalyst credentials and identifiers are present."""
        req_keys = [
            settings.catalyst_project_id,
            settings.catalyst_folder_id,
            settings.catalyst_client_id,
            settings.catalyst_client_secret,
            settings.catalyst_refresh_token,
        ]
        return all(k and k.strip() for k in req_keys)

    # -------------------------------------------------------------------------
    # OAuth2 Token Management
    # -------------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """
        Retrieves a valid OAuth access token for Zoho Catalyst.
        Caches the token and automatically refreshes it before expiration.
        """
        now = time.time()
        # Return cached token if valid (with 60-second safety buffer)
        if self._access_token and now < (self._token_expires_at - 60):
            return self._access_token

        if not self.is_catalyst_configured():
            raise ValueError("Zoho Catalyst credentials are not fully configured in .env")

        accounts_url = settings.catalyst_accounts_url.rstrip("/")
        token_endpoint = f"{accounts_url}/oauth/v2/token"

        params = {
            "refresh_token": settings.catalyst_refresh_token.strip(),
            "client_id": settings.catalyst_client_id.strip(),
            "client_secret": settings.catalyst_client_secret.strip(),
            "grant_type": "refresh_token",
        }

        logger.info("Refreshing Zoho Catalyst OAuth2 access token...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(token_endpoint, params=params)
            if resp.status_code != 200:
                logger.error(f"Zoho OAuth token refresh failed ({resp.status_code}): {resp.text}")
                raise RuntimeError(f"Zoho OAuth token error ({resp.status_code}): {resp.text}")

            data = resp.json()
            if "access_token" not in data:
                err_msg = data.get("error", "Unknown OAuth error")
                logger.error(f"Zoho OAuth error in response: {data}")
                raise RuntimeError(f"Zoho OAuth error: {err_msg}")

            self._access_token = data["access_token"]
            expires_in = int(data.get("expires_in", 3600))
            self._token_expires_at = now + expires_in
            logger.info(f"✅ Zoho Catalyst access token refreshed successfully (expires in {expires_in}s)")
            return self._access_token

    def _get_env_name(self) -> str:
        """
        Returns properly-capitalized Zoho Catalyst Environment header ('Development' or 'Production').
        Catalyst API strictly rejects lowercase 'development'.
        """
        raw = (settings.catalyst_environment or "Development").strip()
        if raw.lower() in ("development", "dev"):
            return "Development"
        if raw.lower() in ("production", "prod"):
            return "Production"
        return raw.capitalize()

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    async def save_file(
        self,
        file_content: bytes,
        filename: str,
        doc_id: str,
    ) -> Dict[str, Any]:
        """
        Saves an uploaded file locally and (if configured) pushes it to Zoho Catalyst File Store.
        Returns storage metadata including local path, cloud status, and file IDs.
        """
        ext = os.path.splitext(filename)[1].lower()
        os.makedirs(settings.upload_dir, exist_ok=True)
        local_path = os.path.join(settings.upload_dir, f"{doc_id}{ext}")

        # 1. Always persist a local copy for instant zero-latency RAG parsing
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(file_content)

        file_size = len(file_content)
        logger.info(f"Local file saved: {local_path} ({file_size:,} bytes)")

        # 2. Upload to Zoho Catalyst File Store if credentials are configured
        catalyst_metadata: Dict[str, Any] = {
            "storage_type": "local",
            "local_path": local_path,
            "filename": filename,
            "file_size": file_size,
            "catalyst_file_id": None,
            "catalyst_folder_id": None,
            "catalyst_project_id": None,
            "catalyst_download_url": None,
        }

        if not self.is_catalyst_configured():
            logger.info("ℹ️ Zoho Catalyst is not configured in .env. Storing file in local disk storage.")
            return catalyst_metadata

        try:
            token = await self._get_access_token()
            api_domain = settings.catalyst_api_domain.strip().rstrip("/")
            project_id = settings.catalyst_project_id.strip()
            folder_id = settings.catalyst_folder_id.strip()
            env_name = self._get_env_name()

            upload_url = f"https://{api_domain}/baas/v1/project/{project_id}/folder/{folder_id}/file"
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            headers = {
                "Authorization": f"Zoho-oauthtoken {token}",
                "Environment": env_name,
            }

            files = {
                "code": (filename, file_content, content_type),
            }
            data = {
                "file_name": f"{doc_id}_{filename}",
            }

            logger.info(f"Uploading '{filename}' to Zoho Catalyst File Store (Folder ID: {folder_id})...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(upload_url, headers=headers, files=files, data=data)

                if resp.status_code in (200, 201):
                    res_json = resp.json()
                    file_info = res_json.get("data", {})
                    file_id = str(file_info.get("id") or "")
                    logger.success(f"✅ Successfully uploaded to Zoho Catalyst File Store! (File ID: {file_id})")

                    catalyst_metadata.update({
                        "storage_type": "catalyst",
                        "catalyst_file_id": file_id,
                        "catalyst_folder_id": folder_id,
                        "catalyst_project_id": project_id,
                        "catalyst_download_url": f"https://{api_domain}/baas/v1/project/{project_id}/folder/{folder_id}/file/{file_id}/download",
                    })
                else:
                    logger.warning(
                        f"Zoho Catalyst upload returned status {resp.status_code}: {resp.text}. "
                        "File remains securely available on local disk."
                    )

        except Exception as e:
            logger.warning(
                f"Could not upload file to Zoho Catalyst File Store ({e}). "
                "Falling back to local disk storage."
            )

        return catalyst_metadata

    async def download_file(
        self,
        local_path: str,
        catalyst_file_id: Optional[str] = None,
    ) -> bytes:
        """
        Retrieves file contents. Prioritizes local cache, falls back to Zoho Catalyst cloud stream.
        """
        # If local copy exists, return directly
        if os.path.exists(local_path):
            async with aiofiles.open(local_path, "rb") as f:
                return await f.read()

        # If local file is missing but Catalyst file ID is available, download from Catalyst
        if catalyst_file_id and self.is_catalyst_configured():
            try:
                token = await self._get_access_token()
                api_domain = settings.catalyst_api_domain.strip().rstrip("/")
                project_id = settings.catalyst_project_id.strip()
                folder_id = settings.catalyst_folder_id.strip()
                env_name = self._get_env_name()

                url = f"https://{api_domain}/baas/v1/project/{project_id}/folder/{folder_id}/file/{catalyst_file_id}/download"
                headers = {
                    "Authorization": f"Zoho-oauthtoken {token}",
                    "Environment": env_name,
                }

                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        # Re-cache locally
                        async with aiofiles.open(local_path, "wb") as f:
                            await f.write(resp.content)
                        return resp.content
                    else:
                        logger.error(f"Catalyst file download failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"Error downloading file from Catalyst: {e}")

        raise FileNotFoundError(f"File not found locally ({local_path}) or on Zoho Catalyst cloud")

    async def delete_file(
        self,
        local_path: str,
        catalyst_file_id: Optional[str] = None,
    ):
        """Deletes file from local disk and Zoho Catalyst File Store."""
        # 1. Delete local file
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"Deleted local file: {local_path}")
        except Exception as e:
            logger.warning(f"Error removing local file {local_path}: {e}")

        # 2. Delete Catalyst Cloud file
        if catalyst_file_id and self.is_catalyst_configured():
            try:
                token = await self._get_access_token()
                api_domain = settings.catalyst_api_domain.strip().rstrip("/")
                project_id = settings.catalyst_project_id.strip()
                folder_id = settings.catalyst_folder_id.strip()
                env_name = self._get_env_name()

                url = f"https://{api_domain}/baas/v1/project/{project_id}/folder/{folder_id}/file/{catalyst_file_id}"
                headers = {
                    "Authorization": f"Zoho-oauthtoken {token}",
                    "Environment": env_name,
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.delete(url, headers=headers)
                    logger.info(f"Catalyst File Store delete response ({resp.status_code})")
            except Exception as e:
                logger.warning(f"Error deleting file {catalyst_file_id} from Catalyst: {e}")

    async def test_connection(self) -> Dict[str, Any]:
        """Diagnostic utility to test Zoho Catalyst credentials and token validity."""
        if not self.is_catalyst_configured():
            return {
                "status": "unconfigured",
                "message": "Zoho Catalyst credentials are not set in .env. Local storage is active.",
                "details": {
                    "project_id_set": bool(settings.catalyst_project_id),
                    "folder_id_set": bool(settings.catalyst_folder_id),
                    "client_id_set": bool(settings.catalyst_client_id),
                    "client_secret_set": bool(settings.catalyst_client_secret),
                    "refresh_token_set": bool(settings.catalyst_refresh_token),
                }
            }

        try:
            token = await self._get_access_token()
            api_domain = settings.catalyst_api_domain.strip().rstrip("/")
            project_id = settings.catalyst_project_id.strip()
            folder_id = settings.catalyst_folder_id.strip()
            env_name = self._get_env_name()

            return {
                "status": "connected",
                "message": f"Successfully authenticated with Zoho Catalyst File Store ({env_name})!",
                "details": {
                    "project_id": project_id,
                    "folder_id": folder_id,
                    "environment": env_name,
                    "api_domain": api_domain,
                    "token_valid": bool(token),
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection check failed: {str(e)}",
            }


storage_service = StorageService()
