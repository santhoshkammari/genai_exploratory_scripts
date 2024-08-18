import argparse
import json

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
        return binary_image

    @staticmethod
    def process_for_analysis(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return np.where(image == 255, 0, 1)

class TextExtractor:
    @staticmethod
    def get_word_coords(image_path):
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                word = data['text'][i]
                word_bbox = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                words.append({'word': word, 'bbox': word_bbox})
        return words

    @staticmethod
    def extract_text_with_boxes(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        binary_image = ImageProcessor.preprocess_image(image_path)
        h, w = binary_image.shape[:2]
        d = pytesseract.image_to_boxes(image)
        boxes = []
        for line in d.splitlines():
            parts = line.split()
            char = parts[0]
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            boxes.append({
                'char': char,
                'bbox': [x1, h - y2, x2, h - y1]
            })
        return boxes

class CharacterAnalyzer:
    @staticmethod
    def get_char_vector(input_image, bbox=None):
        if bbox is not None:
            input_image = input_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        normalization_factor = 1e9
        bbox_features = [float(b) / normalization_factor for b in bbox] if bbox else [0, 0, 0, 0]
        return np.array([
            CharacterAnalyzer.get_ratio(CharacterAnalyzer.get_eroded_image(input_image, 1)),
            CharacterAnalyzer.get_ratio(CharacterAnalyzer.get_eroded_image(input_image, 2)),
            CharacterAnalyzer.get_ratio(CharacterAnalyzer.get_eroded_image(input_image, 3)),
            *bbox_features
        ])

    @staticmethod
    def get_ratio(input_image):
        ones_count = np.count_nonzero(input_image)
        zeros_count = input_image.size - ones_count
        if ones_count > zeros_count:
            ratio = 0
        else:
            ratio = ones_count / zeros_count if zeros_count != 0 else 0
        return round(ratio, 4)

    @staticmethod
    def get_eroded_image(input_image, kernel_filter=None):
        if input_image.size == 0:
            return input_image
        if kernel_filter is None:
            kernel_filter = 2
        kernel = np.ones((kernel_filter, kernel_filter), np.uint8)
        return cv2.erode(input_image.astype(np.uint8), kernel)

class ClusterAnalyzer:
    @staticmethod
    def perform_clustering(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data[:, :3])
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        labeled_data = np.column_stack((data, labels))
        return labeled_data, centers, labels

    @staticmethod
    def get_bold_filtered_chars(data, n_clusters):
        if n_clusters >= len(data):
            return []
        if not isinstance(data, np.ndarray):
            raise TypeError("data is not numpy ndarray")
        labeled_data, _, _ = ClusterAnalyzer.perform_clustering(data=data, n_clusters=n_clusters)
        bold_label = np.argmin(np.bincount(labeled_data[:, -1].astype(int)))
        bold_chars_data = labeled_data[labeled_data[:, -1] == bold_label]
        return bold_chars_data

class BoldCharacterDetector:
    @staticmethod
    def get_bold_chars(chars_list, image, n_clusters):
        ctoi = {char: idx for idx, char in enumerate(list(Counter(_['char'] for _ in chars_list).keys()))}
        itoc = {i: c for i, c in enumerate(ctoi.keys())}
        vocab_vector = [[] for _ in range(len(ctoi))]

        for x in chars_list:
            _char, _bbox = x['char'], x['bbox']
            vocab_vector[ctoi[_char]].append(list(CharacterAnalyzer.get_char_vector(input_image=image, bbox=_bbox)))

        size_and_list = sorted([(len(vocab_vector[idx]), idx) for idx in range(len(ctoi))])
        bold_chars_filtered_data = []

        for (char_freq, char_idx) in tqdm(size_and_list, desc="Processing characters"):
            if char_freq == 1 or not itoc[char_idx].isalpha():
                continue
            labeled_data = ClusterAnalyzer.get_bold_filtered_chars(np.array(vocab_vector[char_idx]), n_clusters=n_clusters)
            if len(labeled_data):
                bold_chars_filtered_data.extend(labeled_data)
        return bold_chars_filtered_data

class BoldWordDetector:
    @staticmethod
    def is_bold(word_bbox, bold_bboxes, threshold=0.3, padding=0):
        x1, y1, w, h = word_bbox
        x2, y2 = x1 + w, y1 + h
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        chars_intersects = []
        for bbox in bold_bboxes:
            bx1, by1, bx2, by2 = bbox[3:7] * 1000000000
            if (x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1):
                chars_intersects.append([bx1, by1, bx2, by2])

        chars_total_dist = 0
        previous = None
        for current_char in sorted(chars_intersects):
            if previous is None:
                chars_total_dist += current_char[2] - current_char[0]
            else:
                if current_char[0] < previous[2]:
                    chars_total_dist += current_char[2] - previous[2]
                else:
                    chars_total_dist += current_char[2] - current_char[0]
            previous = current_char

        return (chars_total_dist / w) > threshold

    @staticmethod
    def detect_bold_words(words, bold_bboxes, threshold=None, padding=None):
        bold_words = []
        for word in words:
            if BoldWordDetector.is_bold(word['bbox'], bold_bboxes, threshold=threshold, padding=padding):
                bold_words.append(word)
        return bold_words

class Visualizer:
    @staticmethod
    def draw_and_save_bold_words(image_path, bold_words, output_path):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        for word in bold_words:
            x, y, w, h = word['bbox']
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        img.save(output_path)
        print(f"Image with bold words highlighted saved as {output_path}")

class BoldTextDetector:
    def __init__(self, config):
        self.config = config

    def detect(self, image_path):
        image = ImageProcessor.process_for_analysis(image_path)
        words = TextExtractor.get_word_coords(image_path)
        chars_list = TextExtractor.extract_text_with_boxes(image_path)
        bold_chars_filtered_data = BoldCharacterDetector.get_bold_chars(
            chars_list=chars_list,
            image=image,
            n_clusters=self.config['n_clusters']
        )
        bold_words = BoldWordDetector.detect_bold_words(
            words,
            bold_chars_filtered_data,
            threshold=self.config['threshold'],
            padding=self.config['padding']
        )
        return bold_words

def main():
    parser = argparse.ArgumentParser(description="Detect bold text in an image.")
    parser.add_argument("--path", required=True, help="Path to the input image")
    parser.add_argument("--save", action="store_true", help="Save the output image with highlighted bold words")
    args = parser.parse_args()

    config = {
        'n_clusters': 2,
        'threshold': 0.3,
        'padding': 0
    }

    detector = BoldTextDetector(config)
    bold_words = detector.detect(args.path)

    with open("bold_words.json","w") as f:
        json.dump(bold_words,f,indent=2)
    print(f"Image with bold words highlighted Json Saved with bold_words.json")

    if args.save:
        output_path = "output_bold_words.jpg"
        Visualizer.draw_and_save_bold_words(args.path, bold_words, output_path)

if __name__ == "__main__":
    main()