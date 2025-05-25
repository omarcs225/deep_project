# Multi-Class Segmentation from a Single-Class Conditional 3D UNet in TensorFlow
##This repository contains a TensorFlow implementation of a 3D UNet model with dense blocks and class-conditional segmentation, allowing a single model to perform segmentation for multiple classes by conditioning on class-specific input.
###Features
Custom 3D UNet architecture

Dense blocks for enhanced feature reuse

Conditional segmentation using external class indicators

Easily scalable to any number of classes

Designed for volumetric data (e.g., medical imaging)
###Architecture
DenseBlock3D: Implements a dense connection block for 3D volumes.

UNet3D_Dense_Cond: Full encoder-decoder structure with skip connections and conditional input support.
###Conditional Segmentation
The model leverages a conditioning value (cond_values) for each class. It appends a constant tensor (with the class-specific value) to the input at each stage of the network, guiding the segmentation process for the selected class.
