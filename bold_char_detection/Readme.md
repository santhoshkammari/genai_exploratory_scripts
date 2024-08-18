# Bold Text Detector

This Python script detects bold text in images using computer vision and machine learning techniques.

## Features

- Detects bold words in images
- Highlights detected bold words
- Supports command-line arguments for easy use

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- scikit-learn
- pytesseract
- Pillow (PIL)
- tqdm

## Installation

1. Clone this repository or download the `bold_text_detector.py` file.

2. Install the required dependencies:

```
pip install opencv-python numpy scikit-learn pytesseract pillow tqdm
```

3. Ensure you have Tesseract OCR installed on your system. If not, follow the installation instructions for your operating system:
   - For Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - For macOS: `brew install tesseract`
   - For Linux: `sudo apt-get install tesseract-ocr`

## Usage

Run the script from the command line with the following syntax:

```
python bold_text_detector.py --path /path/to/your/image.jpg
```

To save the output image with highlighted bold words, add the `--save` flag:

```
python bold_text_detector.py --path /path/to/your/image.jpg --save
```

This will create an output image named `output_bold_words.jpg` in the same directory as the script.

## Output

The script will print the detected bold words and their bounding boxes to the console. If the `--save` flag is used, it will also generate an output image with the bold words highlighted.

## Limitations

- The accuracy of bold text detection may vary depending on the image quality and complexity.
- The script assumes that bold text is less frequent than regular text in the image.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or encounter any problems.

## License

This project is open-source and available under the MIT License.