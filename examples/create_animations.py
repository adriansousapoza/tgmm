#!/usr/bin/env python3
"""
Example script for creating GMM convergence animations.

This script demonstrates how to create animated visualizations 
of the EM algorithm convergence process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tgmm import GaussianMixture, plot_gmm

def create_gmm_convergence_gif(X, output_path='gmm_convergence.gif', 
                               n_components=3, max_iter=20):
    """
    Create an animated GIF showing GMM convergence.
    
    Parameters
    ----------
    X : array-like
        Input data for fitting
    output_path : str
        Path to save the GIF
    n_components : int
        Number of GMM components
    max_iter : int
        Maximum iterations for animation
    """
    
    # Convert to tensor if needed
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    
    # Store model states at each iteration
    model_states = []
    
    # Custom GMM class that saves state at each iteration
    # (This would require modification of the main GMM class)
    # For now, we'll simulate by fitting multiple times with different max_iter
    
    for i in range(1, max_iter + 1):
        gmm = GaussianMixture(
            n_components=n_components,
            max_iter=i,
            random_state=42,  # Keep same initialization
            verbose=False
        )
        gmm.fit(X_tensor)
        
        # Store the state
        model_states.append({
            'weights': gmm.weights_.clone(),
            'means': gmm.means_.clone(),
            'covariances': gmm.covariances_.clone(),
            'iteration': i,
            'converged': gmm.converged_,
            'log_likelihood': gmm.score(X_tensor)
        })
        
        if gmm.converged_:
            # Fill remaining frames with converged state
            for j in range(i + 1, max_iter + 1):
                state_copy = model_states[-1].copy()
                state_copy['iteration'] = j
                model_states.append(state_copy)
            break
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        
        state = model_states[frame]
        
        # Create temporary GMM with current state
        temp_gmm = GaussianMixture(n_components=n_components)
        temp_gmm.weights_ = state['weights']
        temp_gmm.means_ = state['means'] 
        temp_gmm.covariances_ = state['covariances']
        temp_gmm.fitted_ = True
        temp_gmm.converged_ = state['converged']
        
        # Plot current state
        plot_gmm(
            X, temp_gmm,
            ax=ax,
            color_by_cluster=True,
            show_ellipses=True,
            ellipse_std_devs=[1, 2],
            show_means=True,
            mean_size=50,
            title=f"EM Iteration {state['iteration']} | "
                  f"Log-Likelihood: {state['log_likelihood']:.3f} | "
                  f"Converged: {state['converged']}"
        )
        
        return ax.get_children()
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(model_states),
        interval=500,  # 500ms between frames
        blit=False,
        repeat=True
    )
    
    # Save as GIF
    try:
        anim.save(output_path, writer='pillow', fps=2)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Try installing pillow: pip install pillow")
    
    return anim

def create_comparison_gif(X, output_path='gmm_comparison.gif'):
    """
    Create a GIF comparing different covariance types.
    """
    covariance_types = ['full', 'diag', 'spherical', 'tied_full']
    
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    
    # Fit models with different covariance types
    models = {}
    for cov_type in covariance_types:
        gmm = GaussianMixture(
            n_components=3,
            covariance_type=cov_type,
            random_state=42
        )
        gmm.fit(X_tensor)
        models[cov_type] = gmm
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        
        cov_type = covariance_types[frame % len(covariance_types)]
        gmm = models[cov_type]
        
        plot_gmm(
            X, gmm,
            ax=ax,
            color_by_cluster=True,
            show_ellipses=True,
            ellipse_std_devs=[1, 2],
            title=f"Covariance Type: {cov_type} | "
                  f"Log-Likelihood: {gmm.score(X_tensor):.3f}"
        )
        
        return ax.get_children()
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(covariance_types) * 3,  # Show each type 3 times
        interval=1000,  # 1 second between frames
        blit=False,
        repeat=True
    )
    
    # Save as GIF
    try:
        anim.save(output_path, writer='pillow', fps=1)
        print(f"Comparison animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Try installing pillow: pip install pillow")
    
    return anim

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300),
        np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 300),
        np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 2]], 200)
    ])
    
    print("Creating GMM convergence animation...")
    create_gmm_convergence_gif(X, 'examples/gmm_convergence.gif')
    
    print("Creating covariance comparison animation...")
    create_comparison_gif(X, 'examples/gmm_covariance_comparison.gif')
    
    print("Done! Check the 'examples/' folder for GIF files.")