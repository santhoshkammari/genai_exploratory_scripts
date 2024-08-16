import os
import time
from typing import List, Tuple, Dict, Any

import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text_per_page = {}
        self.image_flag = False
        os.environ["PATH"] += os.pathsep + r'C:\Users\santh\Downloads\Release-24.07.0-0 (1)\poppler-24.07.0\Library\bin'


    def extract_text(self, element: LTTextContainer) -> Tuple[str, List[Any]]:
        line_text = element.get_text()
        line_formats = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                for character in text_line:
                    if isinstance(character, LTChar):
                        line_formats.append(character.fontname)
                        line_formats.append(character.size)
        format_per_line = list(set(line_formats))
        return line_text, format_per_line

    def extract_table(self, page_num: int, table_num: int) -> List[List[str]]:
        with pdfplumber.open(self.pdf_path) as pdf:
            table_page = pdf.pages[page_num]
            table = table_page.extract_tables()[table_num]
        return table

    def table_converter(self, table: List[List[str]]) -> str:
        table_string = ''
        for row in table:
            cleaned_row = [
                item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for
                item in row]
            table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
        return table_string[:-1]

    def is_element_inside_any_table(self, element: Any, page: Any, tables: List[Any]) -> bool:
        x0, y0up, x1, y1up = element.bbox
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for table in tables:
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return True
        return False

    def find_table_for_element(self, element: Any, page: Any, tables: List[Any]) -> int:
        x0, y0up, x1, y1up = element.bbox
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for i, table in enumerate(tables):
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return i
        return None

    def crop_image(self, element: LTFigure, pageObj: Any) -> None:
        [image_left, image_top, image_right, image_bottom] = [element.x0, element.y0, element.x1, element.y1]
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)
        with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
            cropped_pdf_writer.write(cropped_pdf_file)

    def convert_to_images(self, input_file: str) -> None:
        images = convert_from_path(input_file)
        image = images[0]
        output_file = 'PDF_image.png'
        image.save(output_file, 'PNG')

    def image_to_text(self, image_path: str) -> str:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text

    def process_page(self, pagenum: int, page: Any, pdfReader: PyPDF2.PdfReader) -> None:
        pageObj = pdfReader.pages[pagenum]
        page_text = []
        line_format = []
        text_from_images = []
        text_from_tables = []
        page_content = []
        table_in_page = -1

        with pdfplumber.open(self.pdf_path) as pdf:
            page_tables = pdf.pages[pagenum]
            tables = page_tables.find_tables()
            if len(tables) != 0:
                table_in_page = 0

            for table_num in range(len(tables)):
                table = self.extract_table(pagenum, table_num)
                table_string = self.table_converter(table)
                text_from_tables.append(table_string)

        page_elements = sorted([(element.y1, element) for element in page._objs], key=lambda a: a[0], reverse=True)

        for _, element in page_elements:
            if table_in_page != -1 and self.is_element_inside_any_table(element, page, tables):
                table_found = self.find_table_for_element(element, page, tables)
                if table_found == table_in_page and table_found is not None:
                    page_content.append(text_from_tables[table_in_page])
                    page_text.append('table')
                    line_format.append('table')
                    table_in_page += 1
                continue

            if not self.is_element_inside_any_table(element, page, tables):
                if isinstance(element, LTTextContainer):
                    line_text, format_per_line = self.extract_text(element)
                    page_text.append(line_text)
                    line_format.append(format_per_line)
                    page_content.append(line_text)

                if isinstance(element, LTFigure):
                    self.crop_image(element, pageObj)
                    self.convert_to_images('cropped_image.pdf')
                    image_text = self.image_to_text('PDF_image.png')
                    text_from_images.append(image_text)
                    page_content.append(image_text)
                    page_text.append('image')
                    line_format.append('image')
                    self.image_flag = True

        self.text_per_page[f'Page_{pagenum}'] = [page_text, line_format, text_from_images, text_from_tables,
                                                 page_content]

    def extract(self) -> Dict[str, List[Any]]:
        try:
            with open(self.pdf_path, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfReader(pdfFileObj)
                for pagenum, page in enumerate(extract_pages(self.pdf_path)):
                    print(f"Processing page {pagenum + 1}")
                    self.process_page(pagenum, page, pdfReader)
        except Exception as e:
            print(f"An error occurred while processing the PDF: {str(e)}")
            raise
        finally:
            if os.path.exists('cropped_image.pdf'):
                os.remove('cropped_image.pdf')
            if os.path.exists('PDF_image.png'):
                os.remove('PDF_image.png')

        return self.text_per_page


if __name__ == "__main__":
    import time

    pdf_path = 'Example PDF.pdf'

    start_time = time.perf_counter()

    try:
        extractor = PDFExtractor(pdf_path)
        result = extractor.extract()

        print(f"Extraction completed in {time.perf_counter() - start_time:.2f} seconds")
        print(f"Number of pages processed: {len(result)}")
        if result:
            print(
                f"Content of first page: {''.join(result['Page_0'][4])}...")  # Print first 200 characters of full content
        else:
            print("No pages were processed.")

        # Optionally, save the results to a file
        import json

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print("Results saved to results.json")
    except Exception as e:
        print(f"An error occurred: {str(e)}")