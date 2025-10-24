# AutoVision - Classifying Cars using Neural Networks 🚗

## Authors: ✏️
Gloria Duo & Nathan Wong

## Project Description: 📁
This repository contains the final project for CS 171 (Introduction to Machine Learning) taught by Professor Mike Wood at San José State University. The project explores the application of deep learning techniques for car image classification. More specifically, it aims to train a neural network that can identify key visual features of a vehicle—such as its body shape, logo, and bumper contours—and use them to accurately classify the car’s make and model. The ability to perform reliable car classification has wide-ranging applications in traffic surveillance, insurance assessment, and autonomous driving. The project focuses on building and evaluating a convolutional neural network (CNN) from scratch to establish a strong baseline for vehicle recognition.

## Project Outline: 📝
### Data Collection Plan: 📘
*Gloria:*

Our project will primarily need a dataset that consists of car images that have been pre-sorted by make and model. Manual data collection of the entire dataset is unrealistic for the scope of this project as it would require too much dedicated time and effort to amass an image gallery large enough for training a neural network. Instead, we will be relying on datasets that have been pre-assembled by others. There are many online sources for car image collections that have already been organized for this purpose. For example, https://www.kaggle.com/datasets/smlztrkk/car-images-dataset has "87,163 images across 20 different car brands and 86 unique models". The data is already organized into 20 car brand folders, and folder contains subfolders with model labels. This dataset will serve as our primary dataset for training. If the need for more data should ever arise, we would ideally locate supplementary datasets that support our target car brands and models and then merge the images into their appropriate subfolders. 

*Nathan:*
### Model Plan: ⚙️
*Gloria:*

**Data Exploration & Pre-Processing:**

Because our dataset is composed of just images, we won't need to worry about distributions and outliers or missing values. Instead, we need to focus on the amount of data that we have available for each car make and model (so we're aware of class imbalances and take the necessary steps to remediate them) as well as image-specific details; that is, that we will need to ensure that each image is standardized in terms of size, brightness, contrast, and that we also filter out unusable images. We can also perform data augmentation at this stage so that our model has more variability in the dataset to work with. Then, we will split our data into train, validate, and test folders so that they're easier to work with.

**Constructing & Training a CNN:**

Our team will be working with PyTorch to construct a CNN that will train on these images. We will start off with a shallow CNN like in class (i.e., 2 convolutional layers & 2 fully-connected layers). Depending on our image sizes, we may also need to condense the outputs of the convolutional layers into smaller dimensions to save on training time. As we progress with the training process, assuming our losses have plateaued but our model is still underfitting, we will take steps like adding more layers to our CNN to that it can improve its ability to identify finer-grained features. If our model is overfitting, then we can take steps like increasing the dropout rate to remedy the issue. Since the model training phase will be iterative and time-consuming, our team will implement features like checkpointing and early-stopping into our model so that we save our progress and use our time efficiently.

*Nathan:*

**Visualizing and Interpreting Model Runs:**




### Project Timeline: 🗓️

(Project due date is 12/11/2025 @ 1:15 P.M.)

⚠️**NOTE:** Project milestones should be completed BY the timeline's indicated date, NOT on the day of.⚠️

| Date | Day | Task / Milestone |
|------|-----|------------------|
| **10/26/25** | Sunday | Submit **Problem of the Week #9** (Project GitHub README Set-up). |
| **11/2/25** | Sunday | Set up **data preprocessing notebook** and complete **data exploration & preprocessing** steps. |
| **11/8/25** | Saturday | Set up **model construction notebook**. Create a **shallow CNN architecture** & training loop using **PyTorch**. |
| **11/15/25** | Saturday | **Evaluate model run results** & tune **hyperparameters or architecture** as needed. Add concurrent **visualizations**. |
| **11/22/25** | Saturday | Continue **evaluation & tuning**. Add or refine **visualizations** as needed. |
| **11/29/25** | Saturday | Set up & complete **analysis & visualization notebook**. Prepare **presentation slides (Google Slides / PowerPoint)**. |
| **12/2/25** or **12/4/25** | Tuesday / Thursday | **In-class presentations.** |
| **12/10/25** | Wednesday | **Finalize & submit project.** |
