# MLProject
# To derive tight robustness in the 𝑙∞ norm for top-k predictions against Advesarial perturbations using randomized smoothing with Laplacian noise
 We are trying to do , instead of Gaussian noise in randomized smoothing we derive robust radii using Laplacian noise and find tight robust radii in  l_∞ , l_0, l_2, l_1 norm bounded perturbation and compare the results
 
# Steps in our approach
We trained the model with CIFAR-10 dataset using WideResNet model with Laplacian data augmentation with sigma value 0.5 and saved the model under the name of cifar_laplace_0.50 and we saved the checkpoint

We tested the model from the saved checkpoint 

Then we made predictions using smoothed classifier and certify the robust l_0, l_1, l_2, l_∞ radii for these predictions

Then we plotted the figures to draw a comparison between different noises (Laplacian, Gaussian and Uniform) against different norms of adversarial perturbations versus certified radius 

We tried finding top-k accuracies but are facing with errors while running the code so we could not plot the graph for accuracy versus certified radius with k as a parameter

But as mentioned in our novelty we computed the certified robust radii against l_∞ adversarial perturbations with randomized smoothing using Laplacian noise for top-1 predictions

The src folder contains a train.py script which will train the model for a specified noise and level of sigma and save the model checkpoint to directory ckpts

The testing script(src/test.py) will load the model checkpoint and make predictions using the smoothed classifier and ceritfied l_0, l_1, l_2, l_∞ robust radii

Results.ipynb is used for plotting graphs for certified robust radii for different kinds of noises

For top-k predictions, we couldn’t implement it as it was time consuming, and we have to find top-k labels predictions whereit becomes challenging to find the probabilities 𝑝𝑖 exactly. So,we couldn’t try to derive robustness radii for top -k predictions but instead did for top 1 prediction, but we did derive robustness
radii against 𝑙0 , 𝑙1 , 𝑙2 , 𝑙∞ norms of adversarial perturbations and compare those while using randomized smoothing by Gaussian and Laplacian noise as mentioned in our novelty.
