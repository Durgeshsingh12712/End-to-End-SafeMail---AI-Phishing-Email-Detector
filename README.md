# End-to-End-SafeMail---AI-Phishing-Email-Detector

Safe_&_Phishing_email_classifier/
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── constants/
│   │   ├── __init__.py
│   │   └── constants.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── exception/
│   │   ├── __init__.py
│   │   └── exception.py
│   ├── logger/
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/
│   │   ├── __init__.py
│   │   └── common.py   
│   └── config/
│       ├── __init__.py
│       └── configuration.py
├── artifacts/
├── logs/
├── requirements.txt
├── setup.py
├── main.py
└── app.py


🏗️ Architecture:

Modular Design: Separated into components, configure, entities, constants, exceptions, logging,  utils, and pipelines
Configuration Management: YAML-based configuration for easy parameter tuning
Comprehensive Logging: Detailed logging throughout the entire pipeline
Custom Exception Handling: Structured error handling with detailed stack traces

🔧 Core Components:

Data Ingestion: Downloads, extracts, and preprocesses the email dataset
Data Validation: Validates data schema and required columns
Data Transformation: TF-IDF vectorization and train/test splitting
Model Training: Logistic Regression model training
Model Evaluation: Comprehensive metrics calculation and evaluation

🚀 Pipelines:

Training Pipeline: End-to-end training workflow
Prediction Pipeline: Single and batch prediction capabilities

🌐 Web Application:

Flask Web App: User-friendly interface with HTML templates
REST API: RESTful endpoints for integration
Batch Processing: Support for multiple email classification

📊 Key Features:

Real-time Predictions: Instant Phishing/Safe classification
Confidence Scores: Probability-based confidence metrics
Comprehensive Metrics: Accuracy, Precision, Recall, F1-Score, Specificity
Confusion Matrix: Visual representation of model performance
Example Templates: Pre-built Phishing/Safe examples for testing

🛠️ Technical Highlights:

Scikit-learn: TF-IDF vectorization and MultinomialNB
Production Ready: Proper error handling, logging, and configuration
Scalable: Modular design allows easy extension
API Integration: REST endpoints for external integration

