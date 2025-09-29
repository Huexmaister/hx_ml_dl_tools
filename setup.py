from setuptools import setup, find_packages

setup(
    name="hx_ml_dl_tools",
    version="1.0.0",
    author="Daniel Sarabia Torres aka Huexmaister",
    author_email="dsarabiatorres@gmail.com",
    description="Librería desarrollo de modelos de ML y DL",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Huexmaister/hx_ml_dl_tools",
    packages=find_packages(),  # Busca automáticamente todos los paquetes
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "catboost==1.2.8",
        "category_encoders==2.8.1",
        "constants_and_tools==2.2.1",
        "imblearn==0.0",
        "joblib==1.5.2",
        "lightgbm==4.6.0",
        "matplotlib==3.10.6",
        "openpyxl==3.1.5",
        "scikit-learn==1.7.2",
        "shap==0.48.0",
        "tensorflow==2.20.0",
        "xgboost==3.0.5",
    ],
    python_requires=">=3.10",
)
