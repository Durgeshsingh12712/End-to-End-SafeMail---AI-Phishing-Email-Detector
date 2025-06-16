# End-to-End-SafeMail---AI-Phishing-Email-Detector

```bash
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
```

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

```bash
# Push Model to AWS

## 1. Login to AWS console.

## 2. Create IAM user for deployment

#    Create IAM user with AdministratorAccess
#	 with specific access

	1. EC2 access : It is virtual machine

    Create EC2 machine (Ubuntu) & add Security groups 5000 port

	2. ECR: Elastic Container registry to save your docker image in aws


    Run the following command on EC2 machine

```

# 3.Create ECR repo to store/save docker image
- save the URI: ```381509086193.dkr.ecr.us-east-1.amazonaws.com/safemail ```

Note: Do the port mapping to this port:- 8501

```bash
sudo apt-get update -y

sudo apt-get upgrade

#Install Docker

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```


### 4. If AWS cli config failed

```bash

sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

sudo apt update

sudo apt install unzip

sudo unzip awscliv2.zip

sudo ./aws/install

```

## AWS
```bash
aws configure
```

# 5. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 6. Setup github secrets:
```bash
AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

AWS_REGION 

AWS_ECR_LOGIN_URI

ECR_REPOSITORY_NAME

```

Image