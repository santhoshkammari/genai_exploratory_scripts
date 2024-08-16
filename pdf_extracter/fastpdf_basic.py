import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import io

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        os.environ["PATH"] += os.pathsep + r'C:\Users\santh\Downloads\Release-24.07.0-0 (1)\poppler-24.07.0\Library\bin'

    def process(self, max_workers=os.cpu_count()) -> Dict[str, Dict[str, Any]]:
        results = {}
        with fitz.open(self.pdf_path) as doc:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_page, page.number, doc): page.number for page in doc}
                for future in futures:
                    page_num = futures[future]
                    results[f'Page_{page_num}'] = future.result()
        return results

    def process_page(self, page_num: int, doc: fitz.Document) -> Dict[str, Any]:
        page = doc[page_num]
        text = page.get_text("dict")
        blocks = np.array(text["blocks"], dtype=object)
        text_blocks = blocks[np.array([block['type'] for block in blocks]) == 0]

        page_content = self.extract_text_and_format(text_blocks)
        page_content['text_from_images'] = self.extract_and_ocr_images(page)
        page_content['full_content'] = ' '.join(page_content['page_text'] + page_content['text_from_images'])

        return page_content

    @staticmethod
    def extract_text_and_format(text_blocks):
        page_text = []
        line_format = []

        for block in text_blocks:
            for line in block['lines']:
                for span in line['spans']:
                    page_text.append(span['text'])
                    line_format.append({
                        'font': span['font'],
                        'size': span['size'],
                        'color': span['color']
                    })

        return {'page_text': page_text, 'line_format': line_format}

    @staticmethod
    def extract_and_ocr_images(page: fitz.Page) -> List[str]:
        images = page.get_images(full=True)
        if not images:
            return []

        def process_image(img):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(image)

        with ThreadPoolExecutor() as executor:
            return list(executor.map(process_image, images))


def read_pdf_result(json_file: str) -> Dict[str, Any]:
    with open(json_file, 'r') as f:
        return json.load(f)


def main():
    pdf_path = 'Example PDF.pdf'
    start_time = time.perf_counter()
    processor = PDFProcessor(pdf_path)
    result = processor.process()
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")

    with open("results.json", "w") as f:
        json.dump(result, f,indent=4)

    json_file = 'results.json'
    all_pages = read_pdf_result(json_file)
    print("All pages:", len(all_pages))

    print("\nPage 0 content:")
    print(all_pages['Page_0']['full_content'][:200] + "...")  # Print first 200 characters



if __name__ == "__main__":
    main()
