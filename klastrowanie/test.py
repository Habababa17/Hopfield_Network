import os
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_patterns(file_path):
    patterns = pd.read_csv(file_path, header=None).to_numpy()
    return patterns

def train_hopfield(patterns, type = 'heb'):
    neuron_amount = patterns[0].size
    weights = np.zeros((neuron_amount, neuron_amount))

    #heb
    for pattern in patterns:
        weights += np.outer(pattern, pattern)
    np.fill_diagonal(weights, 0)
    return weights / patterns.shape[0]

    #oji TODO

def generate_sample(pattern, flip_probability=0.1):

    sample = pattern.copy()
    mask = np.random.rand(len(pattern)) < flip_probability
    sample[mask] = -sample[mask]
    return sample

def async_update(weights, sample, pattern, timeout = 10000):

    states = [sample]
    current_state = sample
    for iteration in range(timeout): 
        new_state = async_forward(current_state, weights)
        states.append(new_state.copy())
        if np.array_equal(new_state, pattern):
            break
        current_state = new_state
    print(len(states))
    return states

def async_forward(state, weights):
    #update 1 neuron chosen randomly
    index = np.random.randint(len(state))  
    weighted_sum = np.dot(weights[index], state)

    new_state = state.copy()
    new_state[index] = 1 if weighted_sum >= 0 else -1
    return new_state

def plot_iteration(states, size, step=10):
    """
    Plot the states of the system over multiple iterations.
    
    Parameters:
    - states: List of states over iterations (each state is a 1D array of -1 and 1 values)
    - size: The dimensions of the state vector (rows, cols) that needs to be reshaped for display
    - step: The interval between iterations to display (default is every 10th iteration)
    """
    # Number of iterations (the length of the states list)
    num_iterations = len(states)

    # Make sure there's at least one state
    if num_iterations == 0:
        print("No iterations to plot.")
        return

    # Calculate how many iterations we are going to plot
    num_plots = num_iterations // step + (1 if num_iterations % step != 0 else 0)

    # Create a figure with a grid large enough to accommodate the plots
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))
    plt.figure(figsize=(cols * 5, rows * 5))  # Adjust the figure size

    # Loop through the states and plot them
    for i in range(0, num_iterations, step):
        # Reshape the state vector into a 2D matrix based on the 'size' (rows, columns)
        state_matrix = np.reshape(states[i], size)
        
        # Determine the subplot index
        subplot_index = (i // step) + 1
        
        # Plot each state at iteration 'i'
        plt.subplot(rows, cols, subplot_index)  # Set position in a grid
        plt.imshow(state_matrix, cmap='coolwarm', interpolation='nearest')
        plt.title(f"Iteration {i}")
        plt.axis('off')  # Turn off the axis

    # Adjust layout to make it cleaner
    plt.tight_layout()
    plt.show()
def plot_iteration_animation(states, size, step=10):
    """
    Plot the states of the system over multiple iterations as an animation.

    Parameters:
    - states: List of states over iterations (each state is a 1D array of -1 and 1 values)
    - size: The dimensions of the state vector (rows, cols) that needs to be reshaped for display
    - step: The interval between iterations to display (default is every 10th iteration)
    """
    # Number of iterations (the length of the states list)
    num_iterations = len(states)

    # Make sure there's at least one state
    if num_iterations == 0:
        print("No iterations to plot.")
        return

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with the first state
    state_matrix = np.reshape(states[0], size)
    im = ax.imshow(state_matrix, cmap='coolwarm', interpolation='nearest')
    ax.axis('off')  # Turn off the axis
    ax.set_title(f"Iteration 0")

    # Update function for animation
    def update(frame):
        # Reshape the current state to the 2D matrix shape
        state_matrix = np.reshape(states[frame], size)
        im.set_data(state_matrix)  # Update the image data
        ax.set_title(f"Iteration {frame}")  # Update the title to show the iteration number
        return [im]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(0, num_iterations, step), interval=100, blit=True)

    # Display the animation
    plt.show()

def main(file_path, size):
    patterns = load_patterns(file_path)
    
    #training
    weights = train_hopfield(patterns, 'heb')

    sample = generate_sample(patterns[2])

    states = async_update(weights, sample, patterns[2])

    # Plot all the states
    plot_iteration_animation(states, size)

if __name__ == "__main__":

    main(os.getcwd() +'/klastrowanie/large-25x25.csv', (25, 25))