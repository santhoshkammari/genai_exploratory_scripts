import os
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure, LTPage
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

class PDFElement(ABC):
    @abstractmethod
    def extract_content(self) -> str:
        pass

class TextElement(PDFElement):
    def __init__(self, element: LTTextContainer):
        self.element = element

    def extract_content(self) -> str:
        return self.element.get_text()

    def get_format(self) -> List[Any]:
        formats = []
        for text_line in self.element:
            if isinstance(text_line, LTTextContainer):
                for character in text_line:
                    if isinstance(character, LTChar):
                        formats.append(character.fontname)
                        formats.append(character.size)
        return list(set(formats))

class TableElement(PDFElement):
    def __init__(self, pdf_path: str, page_num: int, table_num: int):
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.table_num = table_num

    def extract_content(self) -> str:
        with pdfplumber.open(self.pdf_path) as pdf:
            table_page = pdf.pages[self.page_num]
            table = table_page.extract_tables()[self.table_num]
        return self.table_to_string(table)

    @staticmethod
    def table_to_string(table: List[List[str]]) -> str:
        table_string = ''
        for row in table:
            cleaned_row = [item.replace('\n', ' ') if item and '\n' in item else 'None' if item is None else item for item in row]
            table_string += '|' + '|'.join(cleaned_row) + '|' + '\n'
        return table_string.rstrip()

class ImageElement(PDFElement):
    def __init__(self, element: LTFigure, page_obj: Any):
        self.element = element
        self.page_obj = page_obj

    def extract_content(self) -> str:
        self.crop_image()
        self.convert_to_image()
        text = self.image_to_text()
        self.cleanup()
        return text

    def crop_image(self):
        [image_left, image_top, image_right, image_bottom] = [self.element.x0, self.element.y0, self.element.x1, self.element.y1]
        self.page_obj.mediabox.lower_left = (image_left, image_bottom)
        self.page_obj.mediabox.upper_right = (image_right, image_top)
        with PyPDF2.PdfWriter() as writer:
            writer.add_page(self.page_obj)
            with open('cropped_image.pdf', 'wb') as f:
                writer.write(f)

    @staticmethod
    def convert_to_image():
        images = convert_from_path('cropped_image.pdf')
        images[0].save('PDF_image.png', 'PNG')

    @staticmethod
    def image_to_text() -> str:
        img = Image.open('PDF_image.png')
        return pytesseract.image_to_string(img)

    @staticmethod
    def cleanup():
        os.remove('cropped_image.pdf')
        os.remove('PDF_image.png')

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        os.environ["PATH"] += os.pathsep + r'C:\Users\santh\Downloads\Release-24.07.0-0 (1)\poppler-24.07.0\Library\bin'

    def process(self) -> Dict[str, Dict[str, Any]]:
        text_per_page = {}
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(extract_pages(self.pdf_path)):
                page_content = self.process_page(page, pdf_reader.pages[page_num], page_num)
                text_per_page[f'Page_{page_num}'] = page_content
        return text_per_page

    def process_page(self, pdfminer_page: LTPage, pypdf2_page: Any, page_num: int) -> Dict[str, Any]:
        page_elements = sorted([(element.y1, element) for element in pdfminer_page._objs], key=lambda a: a[0], reverse=True)
        page_content = {
            'page_text': [],
            'line_format': [],
            'text_from_images': [],
            'text_from_tables': [],
            'page_content': []
        }

        tables = self.get_tables(page_num)
        for _, element in page_elements:
            if not self.is_element_inside_any_table(element, pdfminer_page, tables):
                self.process_element(element, pypdf2_page, page_content, tables, page_num, pdfminer_page)

        return page_content

    def get_tables(self, page_num: int) -> List[Any]:
        with pdfplumber.open(self.pdf_path) as pdf:
            return pdf.pages[page_num].find_tables()

    @staticmethod
    def is_element_inside_any_table(element: Any, page: LTPage, tables: List[Any]) -> bool:
        x0, y0, x1, y1 = PDFProcessor.get_element_coordinates(element, page)
        return any(PDFProcessor.is_inside_table(x0, y0, x1, y1, table) for table in tables)

    @staticmethod
    def get_element_coordinates(element: Any, page: LTPage) -> Tuple[float, float, float, float]:
        x0, y0up, x1, y1up = element.bbox
        # pdfminer uses bottom-left as origin, so we need to flip the y-coordinate
        y0 = page.height - y1up
        y1 = page.height - y0up
        return x0, y0, x1, y1

    @staticmethod
    def is_inside_table(x0: float, y0: float, x1: float, y1: float, table: Any) -> bool:
        tx0, ty0, tx1, ty1 = table.bbox
        return tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1

    def process_element(self, element: Any, pypdf2_page: Any, page_content: Dict[str, List], tables: List[Any], page_num: int, pdfminer_page: LTPage):
        if isinstance(element, LTTextContainer):
            text_element = TextElement(element)
            page_content['page_text'].append(text_element.extract_content())
            page_content['line_format'].append(text_element.get_format())
            page_content['page_content'].append(text_element.extract_content())
        elif isinstance(element, LTFigure):
            image_element = ImageElement(element, pypdf2_page)
            image_text = image_element.extract_content()
            page_content['text_from_images'].append(image_text)
            page_content['page_content'].append(image_text)
            page_content['page_text'].append('image')
            page_content['line_format'].append('image')
        elif tables:
            table_index = self.find_table_for_element(element, pdfminer_page, tables)
            if table_index is not None:
                table_element = TableElement(self.pdf_path, page_num, table_index)
                table_text = table_element.extract_content()
                page_content['text_from_tables'].append(table_text)
                page_content['page_content'].append(table_text)
                page_content['page_text'].append('table')
                page_content['line_format'].append('table')

    @staticmethod
    def find_table_for_element(element: Any, page: LTPage, tables: List[Any]) -> int:
        x0, y0, x1, y1 = PDFProcessor.get_element_coordinates(element, page)
        for i, table in enumerate(tables):
            if PDFProcessor.is_inside_table(x0, y0, x1, y1, table):
                return i
        return None

def main():
    pdf_path = '2408.08230v1.pdf'
    start_time = time.perf_counter()
    processor = PDFProcessor(pdf_path)
    result = processor.process()
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time-start_time}")

    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()