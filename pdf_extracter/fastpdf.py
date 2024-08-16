import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF


# Single Responsibility Principle: Each class has a single responsibility

class TextExtractor(ABC):
    @abstractmethod
    def extract(self, element: Any) -> Tuple[str, List[Any]]:
        pass


class PDFTextExtractor(TextExtractor):
    def extract(self, element: LTTextContainer) -> Tuple[str, List[Any]]:
        line_text = element.get_text()
        line_formats = np.array([(char.fontname, char.size) for text_line in element
                                 if isinstance(text_line, LTTextContainer)
                                 for char in text_line if isinstance(char, LTChar)])
        format_per_line = np.unique(line_formats, axis=0)
        return line_text, format_per_line.tolist()


class TableExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: str, page_num: int, table_num: int) -> List[List[str]]:
        pass


class PDFTableExtractor(TableExtractor):
    def extract(self, pdf_path: str, page_num: int, table_num: int) -> List[List[str]]:
        with pdfplumber.open(pdf_path) as pdf:
            table = pdf.pages[page_num].extract_tables()[table_num]
        return self.convert_table(table)

    @staticmethod
    def convert_table(table: List[List[str]]) -> str:
        cleaned_rows = [['None' if cell is None else cell.replace('\n', ' ') for cell in row] for row in table]
        return '\n'.join(['|' + '|'.join(row) + '|' for row in cleaned_rows])


class ImageExtractor(ABC):
    @abstractmethod
    def extract(self, page: Any, element: Any) -> str:
        pass


class PDFImageExtractor(ImageExtractor):
    def extract(self, page: fitz.Page, element: LTFigure) -> str:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=element.bbox)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)


# Open-Closed Principle: The PDFProcessor class is open for extension but closed for modification
class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text_extractor = PDFTextExtractor()
        self.table_extractor = PDFTableExtractor()
        self.image_extractor = PDFImageExtractor()

    def process(self) -> Dict[str, List[Any]]:
        text_per_page = {}
        with fitz.open(self.pdf_path) as doc, PyPDF2.PdfReader(self.pdf_path) as pdf_reader:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._process_page, page, pdf_reader.pages[page.number])
                           for page in doc]
                for future in futures:
                    page_num, page_content = future.result()
                    text_per_page[f'Page_{page_num}'] = page_content
        return text_per_page

    def _process_page(self, fitz_page: fitz.Page, pyPDF_page: PyPDF2.PageObject) -> Tuple[int, List[Any]]:
        page_text, line_format, text_from_images, text_from_tables, page_content = [], [], [], [], []

        # Extract tables
        with pdfplumber.open(self.pdf_path) as pdf:
            tables = pdf.pages[fitz_page.number].find_tables()
            for table_num, _ in enumerate(tables):
                table_string = self.table_extractor.extract(self.pdf_path, fitz_page.number, table_num)
                text_from_tables.append(table_string)

        # Process page elements
        for element in extract_pages(self.pdf_path, page_numbers=[fitz_page.number]).__next__():
            if isinstance(element, LTTextContainer):
                line_text, format_per_line = self.text_extractor.extract(element)
                page_text.append(line_text)
                line_format.append(format_per_line)
                page_content.append(line_text)
            elif isinstance(element, LTFigure):
                image_text = self.image_extractor.extract(fitz_page, element)
                text_from_images.append(image_text)
                page_content.append(image_text)
                page_text.append('image')
                line_format.append('image')

        return fitz_page.number, [page_text, line_format, text_from_images, text_from_tables, page_content]


# Liskov Substitution Principle: Subclasses can be substituted for their base classes
# Interface Segregation Principle: Clients are not forced to depend on methods they do not use
# Dependency Inversion Principle: High-level modules do not depend on low-level modules. Both depend on abstractions.

class PDFContentExtractor:
    def __init__(self, processor: PDFProcessor):
        self.processor = processor

    def extract(self) -> Dict[str, List[Any]]:
        return self.processor.process()


if __name__ == "__main__":
    import time

    pdf_path = 'Example PDF.pdf'

    start_time = time.perf_counter()

    processor = PDFProcessor(pdf_path)
    extractor = PDFContentExtractor(processor)
    result = extractor.extract()

    print(f"Extraction completed in {time.perf_counter() - start_time:.2f} seconds")
    print(f"Number of pages processed: {len(result)}")
    print(f"Content of first page: {result['Page_0'][4][:200]}...")  # Print first 200 characters of full content

    # Optionally, save the results to a file
    import json

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("Results saved to results.json")