import base64
import io
from abc import ABC, abstractmethod
from typing import List, Union
from pdf2image import convert_from_path
from pydantic import BaseModel, FilePath, ValidationError, validator, field_validator
from PIL import Image

import os

os.environ["PATH"] += os.pathsep + r'C:\Users\santh\Downloads\Release-24.07.0-0 (1)\poppler-24.07.0\Library\bin'


class FileConverter(ABC):
    """
    Abstract base class for file conversion.
    """

    def __init__(self, file_path: FilePath):
        self.file_path = file_path

    @abstractmethod
    def convert_to_base64(self) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def extract_images(self) -> List[Image.Image]:
        pass


class ImageFileConverter(FileConverter):
    """
    Concrete class for converting image files to Base64.
    """

    def convert_to_base64(self) -> str:
        with open(self.file_path, "rb") as file:
            file_content = file.read()
            return base64.b64encode(file_content).decode('utf-8')

    def extract_images(self) -> List[Image.Image]:
        # Return the image itself as a list since it's already an image file
        image = Image.open(self.file_path)
        return [image]


class PDFFileConverter(FileConverter):
    """
    Concrete class for converting PDF files to Base64.
    Converts each page of the PDF into an image and then to Base64.
    """

    def convert_to_base64(self) -> List[str]:
        images = self.extract_images()
        base64_images = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(base64_image)
        return base64_images

    def extract_images(self, dpi: int = 200) -> List[Image.Image]:
        # Convert PDF to a list of images with the specified DPI
        return convert_from_path(self.file_path, dpi=dpi)

    def save_images(self, output_dir: str) -> List[str]:
        """
        Save the extracted images to disk and return the list of file paths.
        """
        images = self.extract_images()
        saved_image_paths = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        for idx, image in enumerate(images):
            file_path = f"{output_dir}/page_{idx + 1}.jpg"
            image.save(file_path, "JPEG")
            saved_image_paths.append(file_path)
        return saved_image_paths


class FileConverterFactory(BaseModel):
    """
    Factory to determine the appropriate converter based on file extension.
    Uses Pydantic for validation and configuration.
    """
    file_path: FilePath

    @field_validator('file_path')
    def validate_file_extension(cls, value):
        if not value.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.gif']:
            raise ValueError(f"Unsupported file extension: {value.suffix}")
        return value

    def get_converter(self) -> FileConverter:
        if self.file_path.suffix.lower() == ".pdf":
            return PDFFileConverter(self.file_path)
        else:
            return ImageFileConverter(self.file_path)


