import sys
import pandas as pd
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.utils.utils import load_object

class PredictionPipeline:
    def __init__(self):
        self.model_path = "artifacts/model_trainer/model.pkl"
        self.vectorizer_path = "artifacts/data_transformation/vectorizer.pkl"

    def predict(self, text: str):
        try:
            logger.info("Starting Prediction")

            # Load Model and Vectorizer
            model = load_object(file_path = self.model_path)
            vectorizer = load_object(file_path = self.vectorizer_path)

            # Transformer input text
            text_features = vectorizer.transform([text])

            # Make Prediction
            prediction = model.predict(text_features)
            prediction_proba = model.predict_proba(text_features)

            # Get Result
            if prediction[0] == 1:
                result = "SafeMail (Not Phishing)"
                confidence = prediction_proba[0][1]
            else:
                result = "Phishing"
                confidence = prediction_proba[0][0]

            logger.info(f"Prediction: {result}, Confidence: {confidence:.4f}")
            return {"prediction": result, "confidence": float(confidence)}
        
        except Exception as e:
            logger.error("Prediction Failed")
            raise EmailDetectionException(e, sys)
        
    def batch_predict(self, texts: list):
        try:
            logger.info(f"Starting Batch Prediction for {len(texts)} texts")

            # Load Model and Vectorizer
            model = load_object(file_path = self.model_path)
            vectorizer = load_object(file_path = self.vectorizer_path)

            # Transform input texts
            text_features = vectorizer.transform(texts)

            # Make Predictions
            predictions = model.predict(text_features)
            prediction_probas = model.predict_proba(text_features)

            # Prepare results
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
                if pred == 1:
                    result = "SafeMail (Not Phishing)"
                    confidence = proba[1]
                else:
                    result = "Phishing"
                    confidence = proba[0]
                
                results.append({
                    "text": texts[i],
                    "prediction": result,
                    "confidence": float(confidence)
                })

            logger.info("Batch Prediction Completed")
            return results

        except Exception as e:
                logger.error("Batch Prediction Failed")
                raise EmailDetectionException(e, sys)