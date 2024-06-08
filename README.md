## Experimentals
<!-- find ./ -P -mindepth 1 -type f -name "*.ipynb" -printf x | wc -c -->
<!-- find ./ -mindepth 1 -type f -name "*.ipynb" -not -path "*.ipynb_checkpoints*" -printf x | wc -c -->
Current count of `*.ipynb` files = `553`

### A rough research on Neural Networks and Machine Learning

This repository contains various jupyter notebooks on dissecting neural networks and tinkering with it to understand the inner workings. This repository contains the following topics:

* Function approximation using various types of ANN architectures. ```(\NN_Func_Approx\)```
    1. Convex and Lipschitz constraint NN.
    2. Soft Decision Trees
    3. RBF and Neuron as Cluster + Regression.
    4. Normalizing flows and Invertible Neural Networks
    5. Invex function and Connected Set classifiers. 
    6. Dimension Mixer Model
    7. Spatial Neural Network (and Metrics as Transform)
    8. Dynamic Neural Network and NAS.
    9. PCA and Autoencoders

* Spline (and Piecewise) function approximators.
* GANs and Gaussian Mixture Models.
* Perceptron and Hebbs Learning Rule.
* Neuron simulation with dynamic position.
* ANN Optimization and Constraints.

### Usage
This collection jupter notebooks use general libraries like **numpy**, **matplotlib**, **pytorch** as well as **libraries made from scratch : [mylibrary](https://github.com/tsumansapkota/mylibrary).**

### Message
If there is anything I can help with, please let me know, or raise issue. If anything is helpful to you, please make sure to give credit.

If it is not possible to give credit (eg. closed source project), make sure to get consent from the author (me :: [email](mailto:natokpas@gmail.com) :: [homepage](https://tsumansapkota.github.io/about/)).