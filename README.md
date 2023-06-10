## Gender-Classifier-using-Transfer-Learning
This project is a gender classification system implemented using a deep learning neural network trained using a machine learning technique called
transfer learning. More specifically the pre-trained model that was used to train this network is MobileNetV2.

### 🚀Features:
- Utilizes convolutional neural networks to extract meaningful facial features from images.
- The pre-trained weights of the base model is unchanged while few outer layers are trained.
- Implements image augmentation to increase the sensitivity to subtle changes in images.
- Provides a user-friendly interface to interact with the trained model for gender prediction.

### 📌Technologies and Libraries Used:
- Python
- TensorFlow for image processing and building the model
- Flask for building the user interface
- HTML, CSS, and JavaScript for web-based interfaces

### 📘Installation and Usage:
1. Clone the repository: git clone https://github.com/ro-26/gender-classification-using-machine-learning
2. Create a virtual environment: python -m venv env
3. Activate the virtual environment: env\Scripts\activate
4. Install the required dependencies using pip: pip install -r requirements.txt
5. Set FLASK_APP = app.py
6. flask run
7. Access the user interface through a web browser: http://localhost:5000

### 👨‍💻License:
This project is licensed under the MIT License. You are free to modify and use the code as per the terms and conditions mentioned in the LICENSE file.

### ⛔Disclaimer:
This project is intended for educational and demonstration purposes only. It is not designed for high-stakes applications and may not be 100% accurate in gender classification.

For more details and implementation examples, please refer to the documentation and code examples in the repository.


