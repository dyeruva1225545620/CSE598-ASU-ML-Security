# MLProject
# To derive tight robustness in the 𝑙∞ norm for top-k predictions against Advesarial perturbations using randomized smoothing with Laplacian noise
 We are trying to do , instead of Gaussian noise in randomized smoothing we derive robust radii using Laplacian noise and find tight robust radii in  l_∞ , l_0, l_2, l_1 norm bounded perturbation and compare the results
 
# Steps in our approach
We trained the model with CIFAR-10 dataset using WideResNet model with Laplacian data augmentation with sigma value 0.5 and saved the model under the name of cifar_laplace_0.50 and we saved the checkpoint

We tested the model from the saved checkpoint 

Then we made predictions using smoothed classifier and certify the robust l_0, l_1, l_2, l_∞ radii for these predictions

Then we plotted the figures to draw a comparison between different noises (Laplacian, Gaussian and Uniform) against different norms of adversarial perturbations versus certified radius 

We then plotted the top-1 accuracies versus certified radius for different norms of adversarial perturbations for different noises 

We tried finding top-k accuracies but are facing with errors while running the code so we could not plot the graph for accuracy versus certified radius with k as a parameter

But as mentioned in our novelty we computed the certified robust radii against l_∞ adversarial perturbations with randomized smoothing using Laplacian noise for top-1 predictions

The scripts folder contains a train.py script which will train the model for a specified noise and level of sigma and save the model checkpoint to directory ckpts

The testing script(scripts/test.py) will load the model checkpoint and make predictions using the smoothed classifier and ceritfied l_0, l_1, l_2, l_∞ robust radii

Results.ipynb is used for plotting graph 
