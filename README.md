# **Facial Emotion Recognition**

## **Objective**
The goal of this project is to develop a model to recognize emotions (e.g., happy, sad, angry) from facial images using the **FER 2013 dataset** and advanced deep learning techniques such as transfer learning.

---

## **Project Steps**

1. **Preprocess the FER 2013 Dataset**  
   - Resized, normalized and augmented the dataset to improve model generalization.

2. **Transfer Learning**  
   - Used pre-trained CNN models (**VGG16/ResNet**) as feature extractors for emotion recognition.

3. **Fine-Tuning**  
   - Fine-tuned the model to optimize performance for the FER 2013 dataset.

4. **Metrics Reporting**  
   - Evaluated the model using accuracy and F1-scores for different emotions.

---

## **Techniques**
- **Transfer Learning**: Leveraged pre-trained models like VGG16 and ResNet.
- **Data Augmentation**: Applied transformations such as rotation, flipping, and shifting to improve dataset robustness.

---

## **Results**
- **Test Accuracy**: 23.29%  
- "Happy" was the best-recognized emotion with an F1-score of **0.39**. Other emotions like "disgust" and "sad" had near-zero F1-scores.  
- Results indicate the need for a more balanced dataset and specialized models for improved performance.

---

## **Requirements**

### **Hardware Requirements**
- A machine with a GPU (e.g., NVIDIA CUDA-enabled GPU) for faster training.
- At least 8GB of RAM and 10GB of free disk space for the dataset and model.

### **Software Requirements**
- **Python 3.8+**
- Required libraries:
  ```bash
  pip install tensorflow numpy matplotlib scikit-learn kagglehub
  ```

### **Dataset**
- The **FER 2013 dataset** is automatically downloaded using `kagglehub`. Ensure you have a Kaggle API token set up on your machine.

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
  

4. View evaluation metrics and classification results:
   - The model reports accuracy and F1-scores for each emotion after training.

---

## **Future Improvements**
- **Address Dataset Imbalance**: Use oversampling, class weighting, or GANs to balance the dataset.
- **Specialized Models**: Explore models like **FaceNet** or other facial expression-focused architectures.
- **Contextual Features**: Integrate facial landmarks or additional contextual information for better recognition.

---

## **Acknowledgments**
- **Dataset**: [FER 2013](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Pre-trained Models**: [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) and [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function)

---

