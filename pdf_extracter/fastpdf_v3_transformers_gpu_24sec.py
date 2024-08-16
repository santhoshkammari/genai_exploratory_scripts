import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Union
import io

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from tqdm import tqdm

class AdvancedPDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def process(self) -> Dict[str, Dict[str, Any]]:
        text_per_page = {}
        with fitz.open(self.pdf_path) as doc:
            num_pages = len(doc)
            all_images = []
            for page_num in range(num_pages):
                page = doc[page_num]
                page_content = self.process_page_text_and_tables(page)
                images = self.extract_images(page)
                all_images.extend([(page_num, img) for img in images])
                text_per_page[f'Page_{page_num}'] = page_content

            # Batch process all images
            image_texts = self.ocr_images_batch([img for _, img in all_images])
            for (page_num, _), text in zip(all_images, image_texts):
                text_per_page[f'Page_{page_num}']['text_from_images'].append(text)
                text_per_page[f'Page_{page_num}']['page_content'].append(text)

        return text_per_page

    def process_page_text_and_tables(self, page: fitz.Page) -> Dict[str, Any]:
        page_content = {
            'page_text': [],
            'line_format': [],
            'text_from_images': [],
            'text_from_tables': [],
            'page_content': []
        }

        # Extract text and format
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        page_content['page_text'].append(text)
                        page_content['line_format'].append({
                            'font': span['font'],
                            'size': span['size'],
                            'color': span['color']
                        })
                        page_content['page_content'].append(text)

        # Extract tables
        tables = self.extract_tables(page)
        page_content['text_from_tables'] = tables
        page_content['page_content'].extend(tables)

        return page_content

    @staticmethod
    def extract_tables(page: fitz.Page) -> List[str]:
        tables = []
        cells = page.find_tables()
        for table in cells:
            df = pd.DataFrame(table.extract())
            tables.append(df.to_csv(index=False, header=False))
        return tables

    @staticmethod
    def extract_images(page: fitz.Page) -> List[Image.Image]:
        images = []
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            images.append(image)
        return images

    @torch.no_grad()
    def ocr_images_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[str]:
        all_texts = []
        for i in tqdm(range(0, len(images), batch_size), desc="OCR Processing"):
            batch = images[i:i+batch_size]
            pixel_values = self.processor(batch, return_tensors="pt", padding=True).pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, max_length=64)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(generated_text)
        return all_texts

def read_pdf_result(json_file: str, pages: Union[int, List[int], None] = None) -> Dict[str, Any]:
    with open(json_file, 'r') as f:
        data = json.load(f)

    def get_page_content(page_num: int) -> Dict[str, Any]:
        page_key = f'Page_{page_num}'
        if page_key not in data:
            return {}
        return {
            'text': ' '.join(data[page_key]['page_text']),
            'tables': data[page_key]['text_from_tables'],
            'images': data[page_key]['text_from_images'],
            'full_content': ' '.join(data[page_key]['page_content'])
        }

    if pages is None:
        return {f'Page_{i}': get_page_content(i) for i in range(len(data))}
    elif isinstance(pages, int):
        return {f'Page_{pages}': get_page_content(pages)}
    elif isinstance(pages, list):
        return {f'Page_{i}': get_page_content(i) for i in pages if f'Page_{i}' in data}
    else:
        raise ValueError("'pages' must be None, an integer, or a list of integers")

def main():
    pdf_path = '2408.08230v1.pdf'
    start_time = time.perf_counter()
    processor = AdvancedPDFProcessor(pdf_path)
    result = processor.process()
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time}")

    with open("results.json", "w") as f:
        json.dump(result, f)

    json_file = 'results.json'

    all_pages = read_pdf_result(json_file)
    print("All pages:", len(all_pages))

    page_0 = read_pdf_result(json_file, pages=0)
    print("\nPage 0 content:")
    print(page_0['Page_0']['full_content'][:200] + "...")  # Print first 200 characters

if __name__ == "__main__":
    main()