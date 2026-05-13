import time
from pageindex import PageIndexClient
from utils.logger import app_logger

class DocumentService:
    def __init__(self, api_key: str):
        self.client = PageIndexClient(api_key=api_key)

    def process_pdf(self, file_path: str) -> list:
        try:
            result = self.client.submit_document(file_path)
            doc_id = result.get("doc_id")
            
            while True:
                status_res = self.client.get_document(doc_id)
                status = status_res.get("status")
                
                if status == "completed":
                    break
                elif status == "failed":
                    raise Exception("فشلت عملية تحليل المستند في الخادم.")
                
                time.sleep(5)
            
            tree_result = self.client.get_tree(doc_id, node_summary=True)
            return tree_result.get("result", [])
            
        except Exception as e:
            app_logger.error(f"حدث خطأ أثناء معالجة الـ PDF: {str(e)}", exc_info=True)
            raise