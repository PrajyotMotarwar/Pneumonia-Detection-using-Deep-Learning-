Detecting Pneumonia Using Deep Learning
Description
Pneumonia is a severe lung infection that inflames and fills the air sacs in the lungs with fluid or pus. It's a common illness that affects millions of people worldwide. Diagnosing pneumonia involves radiologists analyzing chest X-ray images to detect the presence and severity of the disease. However, studies show that even experienced radiologists often struggle to correctly identify infiltrates on X-rays, leading to delays in diagnosis and increased mortality.

Can AI help detect pneumonia in a way that humans might miss?

Objective
The goal of this project is to build a deep learning model that can accurately detect pneumonia from chest X-ray images.

Methodology
1. Data Collection
Data for this project is sourced from Kaggle: Chest X-Ray Images (Pneumonia), which contains chest X-ray images of pediatric patients (under age 5) from a medical center in Guangzhou, China. The dataset includes 5,856 images with a mix of pneumonia and healthy X-rays.

2. Data Exploration and Splitting
The Kaggle dataset includes a small validation set, so I re-split the data into a custom train-validation-test ratio:

Training Set: 70% of the data
Validation Set: 19% of the data
Test Set: 11% of the data
The training data is imbalanced, with 79% labeled as pneumonia and 21% as healthy.

3. Image Preprocessing
Normalization: Image pixel values were scaled to the range [0, 1] to ensure efficient computation.
Augmentation: To prevent overfitting and increase model robustness, images were augmented.
Grayscale Conversion: Chest X-ray images were converted to grayscale, reducing complexity and focusing on the most important features.
4. Model Architecture
A Convolutional Neural Network (CNN) was used for this binary classification task. The model consists of:

CNN Layers: Extracts features from X-ray images.
Pooling Layers: Reduces dimensionality.
Dropout Layers: Prevents overfitting by randomly disabling some neurons during training.
Fully Connected Layers: Classifies the extracted features into the target classes (pneumonia or healthy).
The model uses ReLU activation for hidden layers and a Sigmoid activation function in the output layer for binary classification.



5. Optimization for AUC Score
Given the critical nature of pneumonia diagnosis, recall is prioritized to minimize false negatives (FN). The model optimizes for AUC score, which is an effective measure of separability in binary classification tasks. The threshold for classification can be adjusted based on the business context to balance false positives (FP) and false negatives (FN).

Results
Model Performance
AUC Score: 97.98% – Our model is able to separate pneumonia images from healthy ones with high accuracy.
Recall: 98.72% – The model has excellent sensitivity, making it reliable for detecting pneumonia cases.
False Positives: 8.49% – A relatively low rate of healthy X-rays being classified as pneumonia.
False Negatives: 0.80% – Very few pneumonia cases are missed, ensuring timely detection.
Training and Evaluation Metrics
Training Loss and Validation Loss: Show that the model does not overfit, with losses converging as training progresses.
Training AUC and Validation AUC: Both scores converge, indicating the model performs well on both training and unseen validation data.
Workflow
To replicate or further explore this project, follow the steps below:

Download Dataset

Get the dataset from Kaggle: Chest X-Ray Pneumonia Dataset
Exploratory Data Analysis (EDA)

Open 01 EDA.ipynb to explore and visualize the dataset.
Model Training and Evaluation

Open 02 CNN Model.ipynb to train the deep learning model, evaluate its performance, and fine-tune as necessary.
Results and Conclusion

Check 03 Results_and_Conclusion.ipynb for performance graphs and final conclusions.
