import os
import sys
from flask import Flask, request, render_template, jsonify
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Initialize Prediction Pipeline
try:
    prediction_pipeline = PredictionPipeline()
    logger.info("Prediction Pipeline Initialized Successfully")

except Exception as e:
    logger.error(f"Failed to intialize Prediction Pipeline: {e}")
    prediction_pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        if prediction_pipeline is None:
            return jsonify({'error': 'Model not Loaded Properly'}), 500
        
        # Get Text from From
        text = request.form.get('email_text', '')

        if not text.strip():
            return jsonify({'error': 'Please enter email text'}), 400
        
        # Make Prediction
        result = prediction_pipeline.predict(text)

        return render_template(
            'result.html',
            text=text,
            prediction = result['prediction'],
            confidence = f"{result['confidence']:.4f}"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction Failed'}), 500
    
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if prediction_pipeline is None:
            return jsonify({'error': 'Model not loaded Properly'}), 500
        
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text field'}), 400
        
        text = data['text']

        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Make Prediction
        result = prediction_pipeline.predict(text)

        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/batch_predict', methods = ['POST'])
def api_batch_predict():
    try:
        if prediction_pipeline is  None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': 'Please Provide Texts field'}), 400

        texts = data['texts']

        if not isinstance(texts, list) or not texts:
            return jsonify({'error': 'texts must be a non-empty list'}),400

        # Make Batch Prediction
        results = prediction_pipeline.batch_predict(texts)

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"API batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__== "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)        