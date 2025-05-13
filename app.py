from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Import your PCA function from another file named pca_utils.py
# Make sure that file exists and has a function named `run_pca(df)`
from pca_utils import run_pca

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)

        # Optional: Validate DataFrame shape/content here

        result = run_pca(df)  # Should return a dict or list, not a DataFrame or numpy array
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Shows full error in terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
