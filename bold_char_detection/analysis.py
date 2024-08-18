import json
import pickle
from collections import Counter
from itertools import chain
from typing import Union, Dict, List

import cv2
import numpy as np
import pytesseract
from PIL import Image
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from umap import UMAP
from PIL import Image, ImageDraw

def get_ratio(input_image,return_type = None):
    # Count the number of ones
    ones_count = np.count_nonzero(input_image)

    # Count the number of zeros
    zeros_count = input_image.size - ones_count

    # Calculate the ratio of ones to zeros
    if ones_count > zeros_count:
        ratio = 0
    else:
        ratio = ones_count / zeros_count if zeros_count != 0 else 0

    if return_type == "all":
        return (ones_count, zeros_count, round(ratio, 4))
    return round(ratio, 4)


def print_img(img, s=None, e=None):
    if s == None: s = 0
    for row in img:
        print("[", end="")
        for i, pixel in enumerate(row[s:e] if e else row[s:]):
            if i > 0:
                print(" ", end="")
            print(f"{'#' if pixel else '*'}", end="")
        print("]")


# Perform erosion on normal_g and bold_g
def get_eroded_image(input_image, kernel_filter: Union[int, None] = None):
    if input_image.size==0:
        return input_image
    if kernel_filter is None:
        kernel_filter = 2
    kernel = np.ones((kernel_filter, kernel_filter), np.uint8)
    return cv2.erode(input_image.astype(np.uint8), kernel)



def get_char_vector(input_image,bbox = None):
    if bbox is not None:
        input_image = input_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

    normalization_factor = 1e9
    bbox_features = [float(b) / normalization_factor for b in bbox] if bbox else [0, 0, 0, 0]

    return np.array([
        get_ratio(get_eroded_image(input_image,1)),
        get_ratio(get_eroded_image(input_image,2)),
        get_ratio(get_eroded_image(input_image,3)),
        *bbox_features
    ])


def perform_clustering(data, n_clusters):

    # Create KMeans instance
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model to the data
    kmeans.fit(data[:,:3])

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # indices = np.arange(len(data))
    # Append labels to the original data
    labeled_data = np.column_stack((data,labels))

    return labeled_data, centers,labels


# Function to get data points for a specific cluster
def get_cluster_data(labeled_data, cluster_number):

    return labeled_data[labeled_data[:, -1] == cluster_number]


def determine_optimal_clusters(data, max_clusters):

    n_samples = len(data)
    max_clusters = min(max_clusters, n_samples - 1)  # Ensure max_clusters is valid
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        # Only calculate silhouette score if we have more than one cluster
        if k > 1 and k < n_samples:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(-1)  # Invalid score

    # Plot the elbow curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot the silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()

    # Determine the optimal number of clusters
    valid_scores = [score for score in silhouette_scores if score != -1]
    if valid_scores:
        optimal_clusters = silhouette_scores.index(max(valid_scores)) + 2
    else:
        optimal_clusters = 2  # Default to 2 clusters if no valid scores

    return optimal_clusters
def plot_clusters_3d(data, labels, centers):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot of the data points
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50, alpha=0.8)

    # Plot the cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')

    # Add labels for each data point
    for i, (x, y, z) in enumerate(data):
        ax.text(x, y, z, f'Point {i}', fontsize=8, alpha=0.7)

    # Customize the plot
    ax.set_title('K-means Clustering Results (3D)', fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_zlabel('Feature 3', fontsize=12)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Label', fontsize=10)

    # Add a legend
    ax.legend(fontsize=10)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_clusters(data, labels, centers):
    plt.figure(figsize=(12, 8))

    # Create a scatter plot of the data points
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')

    # Add labels for each data point
    for i, (x, y) in enumerate(data):
        plt.annotate(f'Point {i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    # Customize the plot
    plt.title('K-means Clustering Results', fontsize=16)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Label', fontsize=10)

    # Add a legend
    plt.legend(fontsize=10)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_umap_auto_clusters(data, max_clusters=10):
    # Perform UMAP dimensionality reduction
    umap_2d = UMAP(n_components=2, random_state=42)
    data_2d = umap_2d.fit_transform(data)

    # Determine optimal number of clusters
    max_clusters = min(max_clusters, len(data) - 1)
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_2d)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_2d, labels))

    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Perform final clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(data_2d)
    centers = kmeans.cluster_centers_

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Elbow method and Silhouette score
    ax1.plot(range(2, max_clusters + 1), inertias, marker='o', label='Inertia')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()

    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='r', label='Silhouette Score')
    ax1_twin.set_ylabel('Silhouette Score')
    ax1_twin.legend()

    # Plot 2: UMAP projection with clusters
    scatter = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')

    # Add labels for each data point
    for i, (x, y) in enumerate(data_2d):
        ax2.annotate(f'Point {i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    ax2.set_title(f'UMAP Projection of Clusters (Optimal clusters: {optimal_clusters})', fontsize=16)
    ax2.set_xlabel('UMAP 1', fontsize=12)
    ax2.set_ylabel('UMAP 2', fontsize=12)

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Cluster Label', fontsize=10)

    # Add a legend
    ax2.legend(fontsize=10)

    # Add grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return optimal_clusters, labels


def draw_and_save_multi_bboxes(image_path, bbox,output_path):
    image = cv2.imread(image_path)
    # Draw bounding boxes on the image
    # color = [(255,0,0),(0,255,0)] if bold_label else [,(255,0,0)]
    for entry in bbox:
        x1, y1, x2, y2 = map(int,entry[3:7]*1000000000)
        # Draw rectangle around the character
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        # Optionally, put the character label on the image
        # label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
        # cv2.putText(image, char, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save the output image
    cv2.imwrite(output_path, image)


def get_word_coords(image_path):
    image = Image.open(image_path)
    # Perform OCR
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract words and their bounding boxes
    words = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():  # Only consider non-empty words
            word = data['text'][i]
            word_bbox = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            words.append({'word': word, 'bbox': word_bbox})
    return words

def less_frequent_label(arr):
    counts = np.bincount(arr.astype(int))
    return np.argmin(counts)


def get_bold_filtered_chars(data,n_clusters):
    if n_clusters>=len(data): return []
    if not isinstance(data,ndarray):
        raise TypeError("data is not numpy ndarray")
    labeled_data, centers, labels = perform_clustering(data=data,
                                                       n_clusters=n_clusters)
    # plot_clusters_3d(data[:, :3], labels, centers)
    bold_label = less_frequent_label(labeled_data[:, -1])
    bold_chars_data = labeled_data[labeled_data[:, -1] == bold_label]
    return bold_chars_data

def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    return binary_image
def extract_text_with_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Preprocess the image
    binary_image = preprocess_image(image_path)

    # Use pytesseract to get bounding boxes and text
    h, w = binary_image.shape[:2]
    d = pytesseract.image_to_boxes(image)

    # Parse pytesseract output
    boxes = []
    for line in d.splitlines():
        parts = line.split()
        char = parts[0]
        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        boxes.append({
            'char': char,
            'bbox': [x1, h - y2, x2, h - y1]  # Adjust y-coordinates due to the image origin at the top left
        })

    return boxes

def is_bold(word_bbox, bold_bboxes, threshold=0.3, padding=0):
    x1, y1, w, h = word_bbox
    x2, y2 = x1 + w, y1 + h

    # Apply padding
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    bold_area = 0
    chars_intersects = []
    for bbox in bold_bboxes:
        bx1, by1, bx2, by2 = bbox[3:7]*1000000000
        # Check for intersection
        if (x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1):
            chars_intersects.append([bx1,by1,bx2,by2])
            intersection = (min(x2, bx2) - max(x1, bx1)) * (min(y2, by2) - max(y1, by1))
            bold_area += intersection
    chars_total_dist = 0
    previous= None
    for current_char in sorted(chars_intersects):
        if previous is None:
            chars_total_dist+=current_char[2]-current_char[0]
        else:
            if current_char[0]<previous[2]:
                chars_total_dist+=current_char[2]-previous[2]
            else:
                chars_total_dist+=current_char[2]-current_char[0]
        previous= current_char

    return (chars_total_dist / w) > threshold


def detect_bold_words(words, bold_bboxes, image_path, threshold=None, padding=None):
    bold_words = []

    for word in words:
    #     if word['bbox']!=[
    #   301,
    #   1179,
    #   110,
    #   25
    # ]: continue
        if is_bold(word['bbox'], bold_bboxes,threshold=threshold,padding=padding):
            bold_words.append(word)




    return bold_words

def image_processing(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    image = np.where(image == 255, 0, 1)
    return image

def get_bold_chars(chars_list=None, image=None, n_clusters=None):
    ctoi = {char: idx for idx, char in enumerate(list(Counter(_['char'] for _ in chars_list).keys()))}
    itoc = {i: c for i, c in enumerate(ctoi.keys())}
    vocab_vector = [[] for _ in range(len(ctoi))]

    for x in chars_list:
        _char, _bbox = x['char'], x['bbox']
        vocab_vector[ctoi[_char]].append(list(get_char_vector(input_image=image, bbox=_bbox)))

    size_and_list = sorted([(len(vocab_vector[idx]), idx) for idx in range(len(ctoi))])
    bold_chars_filtered_data = []

    for (char_freq, char_idx) in tqdm(size_and_list, desc="Processing characters"):
        if char_freq == 1 or not itoc[char_idx].isalpha(): continue
        labeled_data: ndarray = get_bold_filtered_chars(np.array(vocab_vector[char_idx]),
                                                        n_clusters=n_clusters)
        if len(labeled_data):
            bold_chars_filtered_data.extend(labeled_data)
    return bold_chars_filtered_data

def draw_and_save_bold_words(image_path=None, bold_words=None):
    # Draw bounding boxes around bold words
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for word in bold_words:
        x, y, w, h =  word['bbox']
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    # Save the result
    output_path = "output_bold_words.jpg"
    img.save(output_path)
    print(f"Image with bold words highlighted saved as {output_path}")

def get_bold_words(image_path,config = None):
    image = image_processing(image_path)
    words = get_word_coords(image_path)
    chars_list = extract_text_with_boxes(image_path)
    bold_chars_filtered_data = get_bold_chars(chars_list=chars_list,
                                              image=image,
                                              n_clusters=config['n_clusters'])
    bold_words = detect_bold_words(words, bold_chars_filtered_data, image_path,
                                   threshold=config['threshold'],
                                   padding=config['padding'])
    return bold_words


if __name__ == '__main__':
    image_path = '../custom_pdf/images/page_5.jpg'
    config= {
        'n_clusters':2,
        'threshold':0.3,
        'padding':0
    }
    bold_words = get_bold_words(image_path,config)
    draw_and_save_bold_words(image_path=image_path,
                             bold_words=bold_words)



