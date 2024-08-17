import time

from hotpdf import HotPdf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pdf_file_path = "2408.08230v1.pdf"

# Load pdf file into memory
hotpdf_document = HotPdf(pdf_file_path)
logging.info(f"Loaded PDF: {pdf_file_path}")

# # Alternatively, you can also pass an opened PDF stream to be loaded
# with open(pdf_file_path, "rb") as f:
#    hotpdf_document_2 = HotPdf(f)
#    logging.info("Loaded PDF from file stream")

# You can also merge multiple HotPdf objects to get one single HotPdf object
# Assuming hotpdf1 and hotpdf2 are defined elsewhere
# merged_hotpdf_object = HotPdf.merge_multiple(hotpdfs=[hotpdf1, hotpdf2])
# logging.info("Merged multiple HotPdf objects")

# Get the number of pages
num_pages = len(hotpdf_document.pages)
logging.info(f"Number of pages: {num_pages}")

# # Find text
# text_occurrences = hotpdf_document.find_text("foo")
# logging.info(f"Occurrences of 'foo': {len(text_occurrences)}")

# # Find text and its full span
# text_occurrences_full_span = hotpdf_document.find_text("foo", take_span=True)
# logging.info(f"Occurrences of 'foo' with full span: {len(text_occurrences_full_span)}")

# # Extract text in the region
# text_in_bbox = hotpdf_document.extract_text(
#    x0=0,
#    y0=0,
#    x1=100,
#    y1=10,
#    page=0,
# )
# logging.info(f"Text in bbox: {text_in_bbox[:50]}...")  # Logging first 50 characters

# # Extract spans in the region
# spans_in_bbox = hotpdf_document.extract_spans(
#    x0=0,
#    y0=0,
#    x1=100,
#    y1=10,
#    page=0,
# )
# logging.info(f"Number of spans in bbox: {len(spans_in_bbox)}")

# # Extract spans text in the region
# spans_text_in_bbox = hotpdf_document.extract_spans_text(
#    x0=0,
#    y0=0,
#    x1=100,
#    y1=10,
#    page=0,
# )
# logging.info(f"Spans text in bbox: {spans_text_in_bbox[:50]}...")  # Logging first 50 characters

# Extract full-page text

full_page_text = hotpdf_document.extract_page_text(page=0)

with open("page.txt","w",encoding="utf-8") as f:
   f.write(f'{full_page_text}')
