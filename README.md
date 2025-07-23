
# Sign Language Digits Recognition

This project aims to develop a deep learning model capable of recognizing sign language digits (0-9) from images. The goal is to build an accurate and efficient image classification system for this specific task.


# Dataset

dataset link: [kareemabdelhamed/sign-language-digits-images-dataset](https://www.kaggle.com/datasets/kareemabdelhamed/sign-language-digits-images-dataset)

The dataset used for this project is the **Sign Language Digits Images Dataset** from Kaggle. It contains images of hand gestures representing digits from 0 to 9.

<img width="894" height="504" alt="Sign Language Digit Recognition" src="https://github.com/user-attachments/assets/76dde5b2-1c0d-402a-a2dc-8f72a827b409" />


The dataset was split into training, validation, and testing sets. The approximate number of images in each split is as follows:

- **Training Set:** 150 images per digit class (total 1500 images)
- **Validation Set:** 27 images per digit class (total 270 images)
- **Testing Set:** 27 images per digit class (total 270 images)

The images were loaded and preprocessed using Keras's `ImageDataGenerator`. Specifically, the `preprocessing_function` from `keras.applications.mobilenet` was applied to prepare the images for the MobileNet model.


# Model Architecture

The model used in this project is based on the **MobileNet** architecture, pre-trained on the ImageNet dataset. This pre-trained model serves as a powerful feature extractor.

To adapt MobileNet for the sign language digit recognition task, we fine-tuned the model by adding a custom classification head. The original top layers of the MobileNet model were replaced with a new **Dense layer** with **10 units** (corresponding to the 10 digit classes: 0-9) and a **softmax activation function** for multi-class classification.

During the training process, the internal layers of the pre-trained MobileNet model were **frozen**. This means that their weights were not updated during training. Only the weights of the newly added dense layer were trained. This approach leverages the powerful, pre-learned features from the MobileNet base while allowing the model to learn how to classify the specific sign language digits based on these features.


# Training

The model was trained using the **Adam optimizer** with a learning rate of **0.0001**.

The **loss function** used for training was **categorical crossentropy**, which is suitable for multi-class classification problems.

The model was trained for **30 epochs**.

Training was performed using the `train_batches` data generator, and the performance was monitored on the `val_batches` data generator for validation.

# Results

The model achieved an overall **accuracy of 92.9%** on the test set.

The **confusion matrix** below provides a detailed look at the model's performance for each digit class. The rows represent the true labels, and the columns represent the predicted labels. The diagonal elements show the number of correct predictions for each class, while off-diagonal elements indicate misclassifications.

<img width="498" height="432" alt="Sign Language Digit Recognition (1)" src="https://github.com/user-attachments/assets/84dfe5b3-accf-48f9-8d0d-af43a3661514" />


The **classification report** below presents key metrics including precision, recall, and f1-score for each digit class. Precision indicates the accuracy of positive predictions, recall measures the model's ability to find all positive instances, and the f1-score is the harmonic mean of precision and recall.

<img width="494" height="290" alt="Screenshot 2025-07-23 at 11 36 12 PM" src="https://github.com/user-attachments/assets/aba303f1-fc61-4005-9cd2-bf7cf28e675c" />


The **ROC AUC score (macro-average)** is 0.996. This metric assesses the model's ability to distinguish between classes across various threshold settings, with a score closer to 1 indicating better discriminative power.

# Inference

## Setup:
* create python virtual environment and activate it.
* clone this repo.
* run: ```pip install -r requirements.txt```
* run the inference script: ```python infenece.py```

It runs a gradio app where we can upload hand gesture image and submit to get answer and probability distribution for all the digits.

<img width="1440" height="900" alt="Screenshot 2025-07-24 at 12 37 28 AM" src="https://github.com/user-attachments/assets/ce016625-ca13-497e-9a9a-114fe8c59080" />


# Conclusion

This project successfully developed a deep learning model for recognizing sign language digits using a fine-tuned MobileNet architecture. The model demonstrated strong performance on the test set, achieving a high accuracy of 92.9% and an impressive macro-averaged ROC AUC score of 0.996, indicating its effectiveness in classifying the ten digit classes.

Potential future work could involve exploring data augmentation techniques to increase the diversity of the training data, experimenting with different pre-trained model architectures or custom CNN designs, and expanding the dataset to include a wider range of sign language gestures beyond just digits.

## Summary:

### Data Analysis Key Findings
*   The project utilizes the Sign Language Digits Images Dataset from Kaggle, split into training (1500 images), validation (270 images), and testing (270 images) sets.
*   The model is based on a fine-tuned MobileNet architecture, pre-trained on ImageNet, with a custom dense classification layer of 10 units and softmax activation.
*   The internal layers of the pre-trained MobileNet were frozen during training, with only the weights of the custom classification layer updated.
*   The model was trained for 30 epochs using the Adam optimizer with a learning rate of 0.0001 and categorical crossentropy loss.
*   The model achieved an accuracy of 92.9% and a macro-averaged ROC AUC score of 0.996 on the test set.

### Insights or Next Steps
*   The use of a pre-trained MobileNet with frozen layers effectively leverages transfer learning, allowing the model to achieve good performance on a relatively small dataset by focusing training on the classification head.
*   Future work could include incorporating data augmentation to potentially improve robustness and performance, and exploring other advanced architectures or custom CNNs tailored for gesture recognition.

