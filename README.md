# Human Anomaly Detection in 100% Crime Dataset Using Multilevel Classification (CNN, VGG16, AutoEncoders)

## Project Overview
This project focuses on **Human Anomaly Detection** in video surveillance footage using advanced machine learning and deep learning techniques. Leveraging the **UCF Crime Dataset**, which comprises 14 distinct crime classes, the system is designed to detect and classify human anomalies in real-world scenarios effectively. The project's primary goal is to improve the accuracy and reliability of surveillance systems for detecting criminal activities.

## Key Features
- **Dataset Used**: UCF Crime Dataset, containing real-world surveillance footage categorized into 14 crime types.
- **Multilevel Classification**: A hierarchical approach for anomaly detection and classification to improve precision and accuracy.
- **Deep Learning Models**:
  - **Convolutional Neural Networks (CNNs)**: Applied for feature extraction and classification.
  - **VGG16**: A pre-trained deep learning model fine-tuned for anomaly detection tasks.
  - **Autoencoders**: Used for unsupervised anomaly detection by reconstructing frames and identifying deviations.
- **High Accuracy**: Achieved an overall accuracy exceeding 99% in classifying anomalies across the 14 crime categories.

## Methodology
1. **Data Preprocessing**: 
   - Extracted relevant frames from surveillance videos.
   - Performed data augmentation to enhance model robustness.
   - Normalized input data for efficient model training.

2. **Model Architecture**:
   - **CNNs** were trained to identify spatial patterns and features in video frames.
   - **VGG16** was fine-tuned to classify anomalies with higher accuracy by leveraging transfer learning.
   - **Autoencoders** were employed to detect anomalies by identifying significant reconstruction errors.

3. **Training and Validation**:
   - Split dataset into training, validation, and test sets in 80,20,20.
   - Used cross-validation techniques to prevent overfitting.
   - Monitored training performance using accuracy, precision, recall, and F1-score.

4. **Evaluation**:
   - Measured model performance using metrics like accuracy, confusion matrix, and ROC-AUC curves.
   - Compared results across different models to determine the most effective approach.

## Results
- **Overall Accuracy**: Achieved over 99% accuracy in detecting and classifying human anomalies across 14 crime categories.
- **Model Comparison**:
  - CNN: High performance in basic anomaly detection tasks.
  - VGG16: Superior classification accuracy for complex scenarios.
  - Autoencoders: Excellent at identifying subtle anomalies through unsupervised learning.

## Future Work
- **Research Paper**: Currently preparing a comprehensive research paper detailing the methodology, results, and potential future improvements.
- **Improved Dataset**: Plan to incorporate more diverse datasets for broader applicability.
- **Real-Time Implementation**: Extend the system to handle real-time video feeds for live surveillance applications.
- **Enhanced Models**: Explore advanced architectures such as Vision Transformers (ViT) or hybrid models for better performance.

## Getting Started
### Prerequisites
- Python 3.8 or above
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

### Installation
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and preprocess the UCF Crime Dataset.

