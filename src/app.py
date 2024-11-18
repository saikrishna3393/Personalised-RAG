
import os
from flask import Flask, request, jsonify
from utils import pdf_to_text


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
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = file.filename

        #-------------------------------------------------------------------
        # Save the file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        #--------------------------------------------------------------------

        #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #file.save(file_path)
        try:
            extracted_text = pdf_to_text(filepath)

        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"extracted_text": extracted_text})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
