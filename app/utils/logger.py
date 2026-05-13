import logging
import os

def setup_logger(name="VectorlessRAG"):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) 
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler("app.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

app_logger = setup_logger()