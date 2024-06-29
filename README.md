# Real-Time Recognition of Boxing Head Gestures with IMU-Earables: Machine Learning and Dynamic Time Warping
This repository contains the following:

## BoxingRecognition:
- A utility file, including functions for preprocessing the data, training the models, and evaluating the models.
- Dynamic Time Warping Barycenter Averaging file, with minor adjustments, from https://github.com/fpetitjean/DBA.
- Jupyter notebooks for training and testing the models.

## Data:
- Contains the data used for training and testing the models.
- Each folder corresponds to a different session.
- Each session contains the raw data, as extracted from the OpenEarable dashboard, the labeled data, as extracted from the EdgeML dashboard, and the raw data, but renamed to represent which labels were performed.
- Additionally contains DTW templates (Both randomly selected, and DBA based).
## Client:
- A python client for the OpenEarable device, inspired by the OpenEarable JavaScript and Flutter interfaces.

## Server:
- A flask server that acts as an inference API, which can be used to make predictions on the trained models.

