
import os
from flask import Flask, request, jsonify
from utils import pdf_to_text
from engine import invoke_rag

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'jpeg', 'jpg', 'png', 'doc', 'docx'}
UPLOAD_FOLDER = 'Document'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    text_input = request.form.get('query')
    if not text_input:
        return jsonify({'error': 'query input is required'}), 400

    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        # Save the file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            rag_response = invoke_rag(query=text_input,document=filepath)
            return jsonify({"Answer": rag_response}), 200

        except RuntimeError as e:
            print(e)
            return jsonify({"error": "Internal Server error"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
