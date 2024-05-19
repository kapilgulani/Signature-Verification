
# Signature Verification Project

## Link to our Saved Model 
https://drive.google.com/file/d/1vhKQx_Ica7oNLSOdSrcOKVOBQXdBpZ3r/view?usp=sharing

## Overview
This Signature Verification Project aims to provide a robust solution for authenticating users based on their signature images. Utilizing advanced machine learning models, including YOLO v5, ResNet, and MobileNet v2, the project offers a secure and efficient method for signature verification. The system is designed to compare a user-submitted signature image against a pre-stored original signature in the database, determining the authenticity of the signature in real-time.

## Features
- **User Authentication:** Secure login and signup functionality for users to access the signature verification service.
- **Image Upload:** Users can easily upload their signature images for verification.
- **Model Comparison:** Utilizes YOLO v5, ResNet, and MobileNet v2 models to analyze signatures and select the model providing the highest accuracy.
- **Real-Time Verification:** The system quickly compares the uploaded signature against the database-stored original, offering immediate feedback on authenticity.
- **Database Integration:** Uses MongoDB for efficient data storage and management.

## Technology Stack
- **Frontend:** React
- **Backend:** Python (Flask)
- **Database:** MongoDB
- **Machine Learning Models:** YOLO v5, ResNet, MobileNet v2

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Node.js and npm
- MongoDB

## Usage

1. **Sign Up/Login:** Create a new account or log in to access the signature verification service.
2. **Upload Signature:** Once logged in, upload the image of your signature through the provided interface.
3. **Verification:** The system will process the uploaded signature, comparing it to your original signature stored in the database, and return the verification result.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to discuss proposed changes or enhancements.
