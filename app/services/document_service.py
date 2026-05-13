import os
import uuid
from typing import Optional

import fitz
from pageindex.client import PageIndexClient

class DocumentService:
    def __init__(self, api_key: Optional[str] = None):
        api_keys = []
        if api_key:
            api_keys.append(api_key)

        api_keys.extend(
            [
                os.getenv("PAGEINDEX_API_KEY_1"),
                os.getenv("PAGEINDEX_API_KEY_2"),
                os.getenv("PAGEINDEX_API_KEY_3"),
            ]
        )

        self.api_keys = [key for key in api_keys if key]

    def _extract_images(self, file_path: str) -> dict:
        doc = fitz.open(file_path)
        static_dir = os.path.join(os.getcwd(), "static")
        os.makedirs(static_dir, exist_ok=True)
        page_images = {}

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = []
            for img in page.get_images(full=True):
                xref = img[0]
                extracted = doc.extract_image(xref)
                ext = extracted.get("ext", "png")
                image_bytes = extracted.get("image")
                if not image_bytes:
                    continue
                filename = f"{uuid.uuid4().hex}.{ext}"
                path = os.path.join(static_dir, filename)
                with open(path, "wb") as f:
                    f.write(image_bytes)
                images.append(f"/static/{filename}")

            if images:
                page_images[page_index + 1] = images

        doc.close()
        return page_images

    def _inject_images(self, tree, page_images: dict):
        if isinstance(tree, dict):
            tree = [tree]

        if not isinstance(tree, list):
            return tree

        for node in tree:
            page_index = node.get("page_index") or node.get("page")
            if page_index and page_index in page_images:
                existing = node.get("images", [])
                node["images"] = list(dict.fromkeys(existing + page_images[page_index]))
            if node.get("nodes"):
                self._inject_images(node["nodes"], page_images)

        return tree

    def process_pdf(self, file_path: str) -> list:
        if not self.api_keys:
            raise Exception("No PageIndex API keys configured.")

        last_error = None

        page_images = self._extract_images(file_path)
        
        for key in self.api_keys:
            try:
                client = PageIndexClient(api_key=key)
                tree = client.submit_document(file_path)
                return self._inject_images(tree, page_images)
            except Exception as e:
                error_msg = str(e)
                if "LimitReached" in error_msg:
                    print(f"Key ending with {key[-4:]} reached its limit. Switching to the next key...")
                    last_error = error_msg
                    continue
                else:
                    raise e
        
        raise Exception(f"LimitReached on all available keys. Last error: {last_error}")