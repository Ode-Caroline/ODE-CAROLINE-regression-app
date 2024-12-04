# output prediction
power output prediction using Sklearn, FastAPI and Streamlit app

## Table of contents
- [Description](#description)
- [Requirements](requirements)
- [Getting started](getting started)
  - [1. Train and Save the Model](#1-train-and-save-the-model)
  - [2. Deploy FastAPI](#2-deploy-fastapi)
  - [3. Run Streamlit](#3-Streamlit)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Example Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#license)

## Description
This project provides an API and a Streamlit application for predicting power out put (PE) based on environmental factors. The model uses Linear Regression from Sckit-Learn, trained on features including:

- Ambient Temperature (AT)
- Exhaust Vacuum (V)
- Ambient Pressure (AP)
- Relative Humidity (RH)

The API is deployed using FastAPI, and a Streamlit app provides an interactive interface for users to input values and get predictions.