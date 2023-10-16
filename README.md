# Brain-Tumor-Detection-Using-Transfer-Learning

### Table of Contents

- [Introduction](#introduction)
- [Files Included](#files-included)
- [Code Information](#code-information)
- [Dataset](#dataset)
- [Data Processing Techniques](#data-processing-techniques)
- [Models Implemented](#models-implemented)
- [Optimizers Used](#optimizers-used)
- [Results](#results)
- [References](#references)

## Introduction

This project focuses on employing deep learning techniques for the analysis of images. Initially, attempts were made to enhance the dataset using methods such as data augmentation and image processing techniques like histogram equalization and Canny edge detection. However, these techniques unexpectedly resulted in a decrease in image quality, which subsequently affected the accuracy of the models.

To overcome these challenges, the project involved the development of a custom Convolutional Neural Network (CNN) model from scratch. Additionally, well-established transfer learning models including VGG16, ResNet, and Inception were implemented and evaluated. A thorough comparative analysis was conducted to assess the performance of these models, particularly concerning the intricate characteristics of the dataset's images. The training process incorporated the Adam optimizer, with the inclusion of a single dense layer to optimize the models' learning capabilities. The primary objective of this project is to contribute insights into the effectiveness of various deep learning techniques for analyzing complex image datasets, with the overarching goal of advancing image analysis methodologies in the field.

The project's primary objective is the detection of brain tumors using transfer learning techniques. This entails developing a system that can effectively and accurately identify the presence of brain tumors in medical images. The challenge in this project is to harness the capabilities of pre-trained deep learning models to recognize and classify the intricate patterns and structures associated with various types of brain tumors. By employing transfer learning, we aim to expedite the development process while maintaining a high level of accuracy in tumor detection.

## Files Included

1. Jupyter Notebook: Main code implementation
2. Research Poster: Research poster highlighting key aspects of the project
3. Reference Research Paper: Research paper used as a reference for implementing the code

## Code Information

The provided Jupyter Notebook contains the implementation of various models for image analysis, including both scratch-built and pre-trained models.

### Dataset

The dataset used for this project is stored on Google Drive. The dataset comprises two repositories, labeled 'yes' and 'no', for classification purposes.
Dataset Link: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?rvi=1

### Data Processing Techniques

Several image processing techniques were applied to the dataset, including:
- Data augmentation
- Histogram equalization
- Canny edge detection

However, due to some unforeseen issues, these techniques impacted the image analysis, leading to reduced model accuracy.

### Models Implemented

1. CNN Scratch Model
2. Transfer Learning Models:
    - VGG16
    - ResNet
    - Inception

### Optimizers Used

The Adam optimizer was employed during the training of the models.

### Results

A comparative analysis was performed to evaluate the performance of the different models in which VGG-16 outperformed among the implemented models. The results are presented and discussed in the Jupyter Notebook.

## References

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?rvi=1

The code implementation is based on the concepts and methodologies discussed in the following research paper:

https://www.researchgate.net/publication/345449977_Brain_Tumor_Classification_With_Inception_Network_Based_Deep_Learning_Model_Using_Transfer_Learning?enrichId=rgreq-179e656b944352bd03778330be4bd6e0-XXX&enrichSource=Y292ZXJQYWdlOzM0NTQ0OTk3NztBUzoxMDQzMDA0OTQ1MDgwMzIxQDE2MjU2ODMxNjUyMDA%3D&el=1_x_3&_esc=publicationCoverPdf

For more detailed information, please refer to the Jupyter Notebook provided.