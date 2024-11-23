# *Skin Disease Classification using Hybrid CNNs*  

*Course*: BCSE403L - Digital Image Processing  
*Submitted To*: Dr. Geetha S  

---

## *Team Members*  
- *Saimirra R* - 21BCE1656  
- *Harshini B V* - 21BCE1336  
- *Ulagarchana U* - 21BCE1279

---

## *Important Note*  
The augmented dataset provided in this project contains only the first 500 images. The rest of the dataset and related details can be accessed using the following links:  
- [Link to View Augmented Dataset](https://drive.google.com/file/d/1nYuTSkGg2wG-HzIZPNTtkHOBaqvIFglt/view?usp=sharing)  
- [Dataset Link](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset)


For a complete understanding of our project workflow, please watch our *video demonstration*:  
- [Video Demonstration Link](#https://drive.google.com/file/d/1eS3eioIGB-xtX1EGVe6HSOtukupV3zP0/view?usp=sharing)  

---

## *Project Overview*  

Skin diseases can often be identified by visual patterns, and deep learning offers promising solutions for automated classification. This project utilizes *Hybrid Convolutional Neural Networks (CNNs)* to classify skin diseases with high accuracy by combining the strengths of multiple pre-trained models.  

---

## *Key Highlights*  
- *Data Augmentation and Preprocessing* to enhance model performance and reduce overfitting.  
- Use of *Four Pre-trained Models*: DenseNet121, InceptionV3, VGG16, and MobileNetV2.  
- Development of a *Hybrid Model* combining DenseNet121 and InceptionV3.  
- *Feature Map Visualization* to gain insights into feature extraction.  
- *Comprehensive Dataset: Includes **8 classes of skin infections* categorized into bacterial, fungal, parasitic, and viral infections:  
  -*Bacterial Infections*: cellulitis, impetigo  
  -*Fungal Infections*: athlete’s foot, nail fungus, ringworm  
  -*Parasitic Infections*: cutaneous larva migrans  
  -*Viral Infections*: chickenpox, shingles  
---

## *Contents*  
1. [Data Augmentation and Preprocessing](#1-data-augmentation-and-preprocessing)  
2. [Model Architectures and Results](#2-model-architectures-and-results)  
3. [Hybrid Model Architecture and Results](#3-hybrid-model-architecture-and-results)  
4. [Feature Map Visualization](#4-feature-map-visualization)  
5. [Inferences](#5-inferences)  

---

## *1. Data Augmentation and Preprocessing*  

### *Training Data*  
Applied transformations to increase dataset diversity and reduce overfitting:  
- Rotation, shifting, shearing, zooming, and flipping.  

### *Test Data*  
- Only rescaling applied (pixel values normalized to the range [0, 1]).  

### *Image Generators*  
- *flow_from_directory:* Automatically labels images based on folder structure.  
- *class_mode='categorical':* Suitable for multi-class classification tasks.  

---

## *2. Model Architectures and Results*  

We experimented with four pre-trained CNN architectures:  

| *Model*        | *Training Accuracy* | *Testing Accuracy* |  
|-------------------|-----------------------|-----------------------|  
| MobileNetV2       | *0.9888*           | *0.8841*           |  
| DenseNet121       | *0.9933*           | *0.9313*           |  
| VGG16             | *0.9719*           | *0.8541*           |  
| InceptionV3       | *0.9870*           | *0.9313*           |  

---

## *3. Hybrid Model Architecture and Results*  

### *Architecture*  

1. *Feature Extraction:*  
   - DenseNet121 and InceptionV3 used as frozen base models to extract features.  
   - Outputs processed using *GlobalAveragePooling2D* to reduce dimensions.  

2. *Feature Combination:*  
   - Feature maps from both models concatenated using *Concatenate()*.  

3. *Dense Layers for Classification:*  
   - Dense layer (*1024 neurons, ReLU*) for complex pattern learning.  
   - *Dropout (50%)* to prevent overfitting.  
   - Final output layer with *Softmax Activation* for multi-class classification.  

### *Results*  

| *Model*        | *Training Accuracy* | *Testing Accuracy* |  
|-------------------|-----------------------|-----------------------|  
| *Hybrid Model*  | *0.9920*           | *0.9485*           |  

The Hybrid Model outperformed all individual models, demonstrating its superiority in feature extraction and classification.  

---

## *4. Feature Map Visualization*  

### *Process*  
- A test image is preprocessed using *tf.keras.preprocessing.image.load_img* and normalized.  
- Feature maps are extracted from *convolutional and pooling layers* of the base models.  
- Matplotlib is used to visualize and analyze the feature maps, showcasing how the model learns from input images.  

### *Key Observations*  
- Initial layers captured low-level features (edges, textures).  
- Deeper layers extracted complex patterns and meaningful representations.  

---

## *5. Inferences*  

- *Data Augmentation* and preprocessing significantly improved model performance.  
- Combining features from DenseNet121 and InceptionV3 in a *Hybrid Model* resulted in higher accuracy.  
- Dropout layers effectively reduced overfitting, improving model generalization.  
- Feature map visualizations provided valuable insights into model learning.  

---

## *Acknowledgment*  

We extend our heartfelt gratitude to *Dr. Geetha S* for her guidance and support throughout this project.  

---

## *Let’s transform healthcare with AI-powered solutions!*
