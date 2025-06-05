from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "End-To-End-SafeMail---AI-Phishing-Email-Detector"
AUTHOR_USER_NAME = "Durgeshsingh12712"

setup(
    name= "emailDetector",
    version = '0.1.0',
    author= "Durgesh Singh",
    author_email= "durgeshsingh12712@gmail.com",
    description="A Small Python Package for ML App",
    long_description_content_type=long_description,
    long_description_content="text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages= find_packages(),
    install_requires = []
)