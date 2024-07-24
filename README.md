# ML Security and Fairness (CSE-598)

## Project Title
**Deriving Tight Robustness in the 𝑙∞ Norm for Top-k Predictions Against Adversarial Perturbations Using Randomized Smoothing with Laplacian Noise**

## Project Overview
This project aims to derive robust radii using Laplacian noise instead of Gaussian noise in randomized smoothing. We focus on finding tight robust radii in 𝑙∞, 𝑙0, 𝑙2, and 𝑙1 norm-bounded perturbations and comparing the results.

## Approach
1. **Model Training:**
   - Trained the model on the CIFAR-10 dataset using the WideResNet model.
   - Applied Laplacian data augmentation with a sigma value of 0.5.
   - Saved the trained model as `cifar_laplace_0.50` and stored the checkpoint.

2. **Model Testing:**
   - Loaded the saved checkpoint.
   - Made predictions using a smoothed classifier.
   - Certified the robust radii for 𝑙0, 𝑙1, 𝑙2, and 𝑙∞ norms for these predictions.

3. **Comparison and Results:**
   - Plotted figures to compare different noises (Laplacian, Gaussian, and Uniform) against various norms of adversarial perturbations versus certified radius.
   - Attempted to find top-k accuracies but encountered errors, preventing the plotting of accuracy versus certified radius with k as a parameter.
   - Focused on computing the certified robust radii against 𝑙∞ adversarial perturbations using Laplacian noise for top-1 predictions.

## File Structure
- **src/train.py:** 
  - Script to train the model for a specified noise and sigma level.
  - Saves the model checkpoint to the `ckpts` directory.

- **src/test.py:** 
  - Script to load the model checkpoint.
  - Makes predictions using the smoothed classifier.
  - Certifies 𝑙0, 𝑙1, 𝑙2, and 𝑙∞ robust radii.

- **Results.ipynb:** 
  - Notebook for plotting graphs of certified robust radii for different types of noise.

## Challenges and Limitations
- **Top-k Predictions:** 
  - Encountered difficulties in finding exact probabilities for top-k labels.
  - Could not implement top-k predictions due to time constraints.
  - Focused on deriving robustness radii for top-1 predictions.

## Conclusion
Despite the challenges with top-k predictions, we successfully derived and compared robustness radii against 𝑙0, 𝑙1, 𝑙2, and 𝑙∞ norms of adversarial perturbations. This was achieved using randomized smoothing with both Gaussian and Laplacian noise, emphasizing our project's novelty.
