from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
from pathlib import Path
from data.drive_config import (
    SCOPES,
    FOLDER_ID,
    DATA_DIR,
    get_last_sync_time,
    update_last_sync_time,
)


class DriveSync:
    def __init__(self):
        self._initialize_service()

    def _initialize_service(self):
        try:
            creds_json = os.getenv("GOOGLE_CREDENTIALS")
            if not creds_json:
                raise ValueError("GOOGLE_CREDENTIALS not found in environment")

            credentials = service_account.Credentials.from_service_account_info(
                json.loads(creds_json), scopes=SCOPES
            )
            self.service = build("drive", "v3", credentials=credentials)
            print("Successfully initialized Drive service")

            if not self.verify_folder_access():
                raise ValueError("Could not access the specified Drive folder")

        except Exception as e:
            print(f"Error initializing Drive service: {str(e)}")
            raise

    def verify_folder_access(self):
        try:
            print(f"\nVerifying access to folder: {FOLDER_ID}")

            folder = (
                self.service.files()
                .get(fileId=FOLDER_ID, fields="name,mimeType,capabilities")
                .execute()
            )

            print(f"Found folder: {folder.get('name', 'Unknown')}")
            print(f"Type: {folder.get('mimeType', 'Unknown')}")
            print("Capabilities:")
            for cap, value in folder.get("capabilities", {}).items():
                print(f"- {cap}: {value}")

            return True
        except Exception as e:
            print(f"Error verifying folder: {str(e)}")
            return False

    def check_for_updates(self) -> bool:
        try:
            last_sync = get_last_sync_time()

            query = f"'{FOLDER_ID}' in parents and trashed=false"
            if last_sync:
                query += f" and modifiedTime > '{last_sync}'"

            results = (
                self.service.files()
                .list(q=query, fields="files(id, name, modifiedTime)")
                .execute()
            )

            return len(results.get("files", [])) > 0

        except Exception as e:
            print(f"Error checking for updates: {str(e)}")
            return False

    def sync_files(self) -> int:
        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(exist_ok=True)

            # Get all files in the folder
            print("\nListing files in Drive folder...")
            results = (
                self.service.files()
                .list(
                    q=f"'{FOLDER_ID}' in parents and trashed=false",
                    fields="files(id, name, mimeType, modifiedTime)",
                )
                .execute()
            )

            files = results.get("files", [])
            print(f"Found {len(files)} files in Drive folder")

            if len(files) == 0:
                print("No files found in the folder.")
                return 0

            latest_timestamp = None

            for file in files:
                file_path = DATA_DIR / file["name"]
                print(f"\nProcessing: {file['name']}")
                print(f"File type: {file['mimeType']}")

                try:
                    # Download file
                    request = self.service.files().get_media(fileId=file["id"])
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)

                    print("Downloading...")
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            print(f"Download {int(status.progress() * 100)}%")

                    # Save file
                    fh.seek(0)
                    with open(file_path, "wb") as f:
                        f.write(fh.read())
                    print(f"Saved to {file_path}")

                    # Track latest modification time
                    if not latest_timestamp or file["modifiedTime"] > latest_timestamp:
                        latest_timestamp = file["modifiedTime"]

                except Exception as e:
                    print(f"Error downloading file {file['name']}: {str(e)}")
                    continue

            # Update last sync time
            if latest_timestamp:
                update_last_sync_time(latest_timestamp)
                print(f"\nUpdated last sync time to: {latest_timestamp}")

            return len(files)

        except Exception as e:
            print(f"Error syncing files: {str(e)}")
            print("Stack trace:")
            import traceback

            print(traceback.format_exc())
            return 0
