import pytesseract
import cv2
from .config import TESSERACT_PATH, IMAGE_SIZE, OUTPUT_DIR
from .preprocessing import load_image, preprocess_image
import os

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def detect_words(image):
    """Detects words in the given image using Tesseract OCR."""
    return pytesseract.image_to_string(image), pytesseract.image_to_boxes(image)

def draw_boxes(image, boxes):
    """Draws bounding boxes around detected words."""
    h, w, _ = image.shape
    for box in boxes.splitlines():
        b = box.split()
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(image, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

def save_output(image, output_path):
    """Saves the output image to the specified path."""
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Load and preprocess the image
    image_path = os.path.join('images', 'sample.jpg')
    image = load_image(image_path)
    processed_image = preprocess_image(image)
    
    # Detect words and draw bounding boxes
    detected_text, boxes = detect_words(processed_image)
    print("Detected text:", detected_text)
    draw_boxes(image, boxes)
    
    # Save output
    output_image_path = os.path.join(OUTPUT_DIR, 'output_image.jpg')
    save_output(image, output_image_path)
    print(f"Output saved at {output_image_path}")
