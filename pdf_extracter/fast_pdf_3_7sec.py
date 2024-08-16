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
import pytesseract


class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        os.environ["PATH"] += os.pathsep + r'C:\Users\santh\Downloads\Release-24.07.0-0 (1)\poppler-24.07.0\Library\bin'

    def process(self) -> Dict[str, Dict[str, Any]]:
        text_per_page = {}
        with fitz.open(self.pdf_path) as doc:
            num_pages = len(doc)
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.process_page, self.pdf_path, page_num) for page_num in range(num_pages)]
                for future in as_completed(futures):
                    page_num, page_content = future.result()
                    text_per_page[f'Page_{page_num}'] = page_content
        return text_per_page

    @staticmethod
    def process_page(pdf_path: str, page_num: int) -> Tuple[int, Dict[str, Any]]:
        page_content = {
            'page_text': [],
            'line_format': [],
            'text_from_images': [],
            'text_from_tables': [],
            'page_content': []
        }

        with fitz.open(pdf_path) as doc:
            page = doc[page_num]

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
            tables = PDFProcessor.extract_tables(page)
            page_content['text_from_tables'] = tables
            page_content['page_content'].extend(tables)

            # Extract images and perform OCR
            images = PDFProcessor.extract_images(page)
            for img in images:
                image_text = PDFProcessor.ocr_image(img)
                page_content['text_from_images'].append(image_text)
                page_content['page_content'].append(image_text)

        return page_num, page_content

    @staticmethod
    def extract_tables(page: fitz.Page) -> List[str]:
        tables = []
        cells = page.find_tables()
        for table in cells:
            df = pd.DataFrame(table.extract())
            tables.append(df.to_csv(index=False))
        return tables

    @staticmethod
    def extract_images(page: fitz.Page) -> List[Image.Image]:
        images = []
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
        return images

    @staticmethod
    def ocr_image(image: Image.Image) -> str:
        return pytesseract.image_to_string(image)




def read_pdf_result(json_file: str, pages: Union[int, List[int], None] = None) -> Dict[str, Any]:
    """
    Read and return the processed PDF results from a JSON file.

    Args:
    json_file (str): Path to the JSON file containing the processed PDF results.
    pages (int, List[int], or None): Specify which pages to retrieve.
                                     If None, return results for all pages.
                                     If int, return results for that specific page.
                                     If List[int], return results for the specified pages.

    Returns:
    Dict[str, Any]: A dictionary containing the requested PDF content.
    """
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Function to get content for a single page
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

    # Handle different types of 'pages' input
    if pages is None:
        # Return content for all pages
        return {f'Page_{i}': get_page_content(i) for i in range(len(data))}
    elif isinstance(pages, int):
        # Return content for a single page
        return {f'Page_{pages}': get_page_content(pages)}
    elif isinstance(pages, list):
        # Return content for specified pages
        return {f'Page_{i}': get_page_content(i) for i in pages if f'Page_{i}' in data}
    else:
        raise ValueError("'pages' must be None, an integer, or a list of integers")


# Example usage
def main():
    pdf_path = '2408.08230v1.pdf'
    start_time = time.perf_counter()
    processor = PDFProcessor(pdf_path)
    result = processor.process()
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time}")
    #
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)

    json_file = 'results.json'

    # Read all pages
    all_pages = read_pdf_result(json_file)
    print("All pages:", len(all_pages))

    # Read a specific page (e.g., page 0)
    page_0 = read_pdf_result(json_file, pages=0)
    print("\nPage 0 content:")
    print(page_0['Page_0']['full_content']+ "...")  # Print first 200 characters

    # Read multiple specific pages (e.g., pages 1 and 2)
    # specific_pages = read_pdf_result(json_file, pages=[1, 2])
    # print("\nPages 1 and 2:")
    # for page, content in specific_pages.items():
    #     print(f"{page}:", content['text'] + "...")  # Print first 100 characters of text


if __name__ == "__main__":
    main()