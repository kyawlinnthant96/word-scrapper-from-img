import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh_image = cv2.threshold(grey_image, 150, 255, cv2.THRESH_BINARY)
    return thresh_image