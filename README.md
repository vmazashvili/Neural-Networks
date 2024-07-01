# Neural-Networks
Reimplementation the method from the paper - MaskTune: Mitigating Spurious Correlations by Forcing to Explore 

## Intrdocution 
This project is a reimplementation of MaskTune, a novel technique described in the paper MaskTune: Mitigating Spurious Correlations by Forcing to Explore. This single-epoch finetuning technique addresses the challenge of spurious correlations in over-parametrized deep learning models. Spurious correlations refer to coincidental associations between input and target variables that can lead to poor generalization performance [6]. It forces the model to explore other train variables by concealing the first explored ones, causing the training to ditch its myopic and greedy feature-seeking character, while encouraging exploration, leveraging more input variables.

## Implementation
This project is implemented on a modified MNIST dataset. One base model was trained on this dataset and several fine-tuned ones with several masking approaches. You can access the model checkpoints from the `/checkpoints` directory. To load those checkpoints in the notebook, change the `directory` variable to the desired one, which will modify the root directory location.

The pipeline for the project is as follows: 
1. Prepare the dataset by inducing spurious features
2. Train the ERM model on that dataset
3. Generate saliency maps from the trained model using XGradCAM
4. Based on these activation maps, create masks and apply them to the dataset, generating a new, masked dataset
5. Finetune model for one epoch with the masked dataset

### Dataset Preparation
The appropriate dataset is created from MNIST to illustrate the effectiveness of the technique further. At first, we distinguish between two MNIST digit groups (0-4 and 5-9). Those groups are remapped into class 0 and class 1 respectively. We induce a spurious feature (blue square in the top left corner) to 99% of samples in newly acquired class 0 and 1% of the samples of the new class 1. 
As for testing, raw and modified, biased test sets are used (both of them remapped as well). The modified test set has a spurious feature for only class one.
### The Neural Network
This project uses the same Convolutional Neural Network as the project. One feature of the `SmallCNN` class is a get_grad_cam_target_layer function which will grab the last convolutional layer and use it for the saliency map generation. 
The hyperparameters are the same as suggested by the authors:
```
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
batch_size = 128
epochs = 20
lr_decay_epochs = 25
lr_decay_factor = 0.5
number_of_classes = 2
```

### Masking
Masking function 𝓖 is a key factor in the MaskTune method. It identifies and masks the most discriminative features in the sample found by the fully trained model, thus it is applied offline. This will encourage the model to explore more features during the fine-tuning.


$𝓖:$ masking function, here xGradCAM is used.


*   Input: image $H×W×C$
*   output: localization map $𝓐$ with dimensions $H×W×1$

For each sample $(x_i, y_i)$, $x_i ∈X$ and $y_i∈Y$, the masking is done as following:


*   $𝓐_{x_i}=𝓖(m_θ(x_i), y_i)$
*   $x̂_i=Τ(𝓐_{x_i},τ)⊙x_i$

Where Τ is a thresholding function with the threshold factor τ $(i.e., Τ=𝟙_{𝓐_{x_i}≤τ})$ and ⊙ denotes element-wise multiplication.

$Τ(𝓐_{x_i})$, in our case [8, 8]  is upsampled to match the size of the input [3, 28, 28].



**The steps are the following:**


1.   Learn model $m_θ^{initial}$ using the original data $𝓓^{initial}$
2.   Create masked set $𝓓^{masked}$ using $m_θ^{initial}$, $𝓖$ and $Τ$
3.  $m_θ^{initial}$ is tuned using $𝓓^{masked}$ to obtain $m_θ^{final}$

This project experiments with 3 different masking methods, all of them leverage the saliency maps. 
1. "threshold" method - generates a mask based on a user-defined threshold value; masking the regions where the saliency map has greater values than the threshold.
2. "top_k" - creates a mask based on the user-defined threshold k. It masks the k percentile most activated pixels.
3. "Mean" masking - The mask is created based on a scaled value of the calculated mean of the saliency map. 

### Training and Finetuning 
First, we get the checkpoint for the ERM model, using the cross-entropy loss function and stochastic gradient descent optimizer. The training is done for 50 epochs (Due to limited resources) The learning rate decays after every number of specified epochs. The final learning rate value from ERM training is used as a finetuning hyperparameter later on. 
The models with the different masking methods and parameters are finetuned in the Masking and Finetuning section, where we can define the desired configuration with `method` and `param` variables. Finetuning models one by one enables us to save and load many checkpoints without RAM bottlenecking, to plot the method's effectiveness. 
The fine-tuned model checkpoints are then saved with the appropriate namings, which are later leveraged to plot out their performances 
Each model is tested on raw and biased test datasets. 
## Results
In the Plotting and Visualization section, we can modify the `base_model` and `finetuned_model` variables to output the saliency maps and masks for the desired finetuned model checkpoint. 

![accuracies](https://github.com/vmazashvili/Neural-Networks/assets/36914777/c18242fc-efc8-4111-9d0d-d03e4086a7a1)
From this plot, It is visible that MaskTune is a viable method, able to boost performance significantly. However, the parameters and the masking methods should be right. In our case, Threshold methods with moderately high parameters and top_k methods with small and moderate parameter values performed the best. Mean masking with param=0.9 performed well on the biased set, but poorly on the raw test set. Overall the best performer was top_k_0.1 on both, biased and raw test sets. 
Here are the saliency maps for the worst performing model: Mean masking with 0.1 threshold:
![image](https://github.com/vmazashvili/Neural-Networks/assets/36914777/acc2b72b-41bc-4daf-abbe-0cec219dcebf)

While it masks the spurious features, the model fails to classify non-spurious samples

And the best performing one: top_k_0.1
![image](https://github.com/vmazashvili/Neural-Networks/assets/36914777/63915979-2594-4c48-96f4-4ad490bc8193)

We can tell that this model can generalize better and perform well when encountering samples with spurious features.

## References

1. **Normalization and Masking Techniques:**
    - Taghanaki, S. A., Khani, A., Khani, F., Gholami, A., Tran, L., Mahdavi-Amiri, A., & Hamarneh, G. (2022). MaskTune: Mitigating Spurious Correlations by Forcing to Explore. *arXiv preprint arXiv:2210.00055*. Retrieved from [https://arxiv.org/abs/2210.00055](https://arxiv.org/abs/2210.00055).

2. **Dataset Handling and Masking Function Implementation:**
    - Deng, L. (2012). The MNIST Database of Handwritten Digit Images for Machine Learning Research. *IEEE Signal Processing Magazine, 29*(6), 141-142. Retrieved from [https://ieeexplore.ieee.org/document/6336674](https://ieeexplore.ieee.org/document/6336674).
    - Verma, J. (2020). How to Load and Plot the MNIST Dataset in Python. *AskPython*. Retrieved from [https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python](https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python).

3. **Grad-CAM Implementation and Handling:**
    - Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs. *arXiv preprint arXiv:2008.02312*. Retrieved from [https://arxiv.org/abs/2008.02312](https://arxiv.org/abs/2008.02312).

4. **General PyTorch Implementation and Training Loop:**
    - TensorFlow. (n.d.). Writing a Training Loop from Scratch. Retrieved from [https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch).

5. **Visualization and Plotting:**
    - GeeksforGeeks. (n.d.). Bar Plot in Matplotlib. Retrieved from [https://www.geeksforgeeks.org/bar-plot-in-matplotlib/](https://www.geeksforgeeks.org/bar-plot-in-matplotlib/).

6. **Masking Techniques in Spurious Correlation Mitigation:**
    - Taghanaki, S. A., Khani, A., Khani, F., Gholami, A., Tran, L., Mahdavi-Amiri, A., & Hamarneh, G. (2022). MaskTune: Mitigating Spurious Correlations by Forcing to Explore. *arXiv preprint arXiv:2210.00055*. Retrieved from [https://arxiv.org/abs/2210.00055](https://arxiv.org/abs/2210.00055).

7. **MNIST Dataset Details:**
    - Deng, L. (2012). The MNIST Database of Handwritten Digit Images for Machine Learning Research. *IEEE Signal Processing Magazine, 29*(6), 141-142. Retrieved from [https://ieeexplore.ieee.org/document/6336674](https://ieeexplore.ieee.org/document/6336674).

8. **Loading and Plotting MNIST Dataset:**
    - Verma, J. (2020). How to Load and Plot the MNIST Dataset in Python. *AskPython*. Retrieved from [https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python](https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python).

9. **Writing a Training Loop in TensorFlow:**
    - TensorFlow. (n.d.). Writing a Training Loop from Scratch. TensorFlow Core. Retrieved from [https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch).

10. **Grad-CAM for CNN Visualization:**
    - Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs. *arXiv preprint arXiv:2008.02312*. Retrieved from [https://arxiv.org/abs/2008.02312](https://arxiv.org/abs/2008.02312).

11. **Bar Plot in Matplotlib:**
    - GeeksforGeeks. (n.d.). Bar Plot in Matplotlib. Retrieved from [https://www.geeksforgeeks.org/bar-plot-in-matplotlib/](https://www.geeksforgeeks.org/bar-plot-in-matplotlib/).
