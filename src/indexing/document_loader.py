from pathlib import Path
from typing import Dict, List
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DocumentLoader:
    def __init__(self, data_dir: str = "data"):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if "RENDER" in os.environ:
            self.data_dir = self.project_root / data_dir
            logging.info("Running on Render, using absolute path for data directory.")
        else:
            self.data_dir = self.project_root / data_dir
            logging.info("Running locally, using relative path for data directory.")
        if not self.data_dir.exists():
            logging.error(f"Data directory not found at {self.data_dir}.")
            raise FileNotFoundError(
                f"Data directory not found at {self.data_dir}. Please ensure you have created the 'data' directory in the correct location and added your PDF files to it."
            )

    def load_documents(self) -> Dict[str, List[Document]]:
        documents = [
            f
            for f in os.listdir(self.data_dir)
            if f.endswith((".pdf", ".docx", ".txt", ".xlsx", ".csv"))
        ]
        if not documents:
            logging.warning(f"No PDF files found in {self.data_dir}")
            return {}
        loaded_docs = {}
        logging.info(f"Found {len(documents)} PDF files in {self.data_dir}")
        for file in documents:
            key = os.path.splitext(file)[0]
            file_path = str(self.data_dir / file)
            logging.info(f"Attempting to load: {file_path}")
            try:
                loaded_docs[key] = SimpleDirectoryReader(
                    input_files=[file_path]
                ).load_data()
                logging.info(f"Successfully loaded: {key}")
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        return loaded_docs
