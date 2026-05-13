import os
from typing import Optional
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

    def process_pdf(self, file_path: str) -> list:
        if not self.api_keys:
            raise Exception("No PageIndex API keys configured.")

        last_error = None
        
        for key in self.api_keys:
            try:
                client = PageIndexClient(api_key=key)
                return client.submit_document(file_path)
            except Exception as e:
                error_msg = str(e)
                if "LimitReached" in error_msg:
                    print(f"Key ending with {key[-4:]} reached its limit. Switching to the next key...")
                    last_error = error_msg
                    continue
                else:
                    raise e
        
        raise Exception(f"LimitReached on all available keys. Last error: {last_error}")