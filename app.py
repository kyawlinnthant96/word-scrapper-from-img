from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from src.word_detection import detect_words, draw_boxes, save_output
from src.preprocessing import load_image, preprocess_image
from src.config import OUTPUT_DIR

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return  jsonify({'error': 'No file Upload'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}),400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    file.save(file_path)

    image = load_image(file_path)
    processed_image = preprocess_image(image)
    detected_text,boxes = detect_words(processed_image)
    draw_boxes(image,boxes)

    return jsonify({'detect_text': detected_text})

if __name__ == '__main__':
    app.run(debug=True,port=5000)
