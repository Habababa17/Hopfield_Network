import os
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch

def load_patterns(file_path):
    patterns = pd.read_csv(file_path, header=None).to_numpy()
    return patterns

def train_hopfield(patterns, type = 'heb'):
    neuron_amount = patterns[0].size
    M = patterns.shape[0]
    weights = torch.zeros((neuron_amount, neuron_amount), device='cuda')
    patterns = torch.tensor(patterns, dtype=torch.float32, device='cuda')
    patternc = torch.rand(neuron_amount)
    #heb
    if type == 'heb':
        for pattern in patterns:
              # Create a tensor on the CPU
            patternc = pattern.to('cuda')
            weights += torch.ger(patternc, patternc)
        weights = weights.cpu().numpy()
        np.fill_diagonal(weights, 0)
        return weights/ M
    #oja
    elif type == 'oja':
        learning_rate = 0.001
        for pattern in patterns:
            y = np.dot(weights, pattern)
            for i in range(neuron_amount):
                for j in range(neuron_amount):
                    if i != j:
                        weights[i, j] += (learning_rate / M) * y[i] * (pattern[j] - weights[i, j] * y[i])
        # for i in range(50):
        #     for pattern in patterns:
               
        #         y = np.dot(weights, pattern)  
        #         weights += learning_rate * (np.outer(y, pattern) - np.outer(y, y) * weights) / M
        
        return weights
    else:
        raise ValueError("Invalid type. Use 'heb' for Hebbian or 'oja' for Oja's rule.")

def generate_sample(pattern, flip_probability=0.1):

    sample = pattern.copy()
    mask = np.random.rand(len(pattern)) < flip_probability
    sample[mask] = -sample[mask]
    return sample

def is_stable(sample, weights):
    return (sync_forward(sample, weights) == sample).all()

def async_update(weights, sample, patterns, timeout = 50000):
    states = [sample]
    current_state = sample
    for iteration in range(timeout): 
        new_state = async_forward(current_state, weights)
        states.append(new_state.copy())
        if iteration%100==0:
            #print(iteration)
            if is_stable(current_state, weights):
                break
        current_state = new_state
    #print(len(states))
    return states

def async_forward_GPU(state, weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    index = torch.randint(0, len(state), (1,), device=device).item()
    weighted_sum = torch.dot(weights_tensor[index], state_tensor)

    new_state = state_tensor.clone()
    new_state[index] = 1 if weighted_sum >= 0 else -1

    return new_state.cpu().numpy()
def async_forward(state, weights):
    #update 1 neuron chosen randomly
    index = np.random.randint(len(state))  
    weighted_sum = np.dot(weights[index], state)

    new_state = state.copy()
    new_state[index] = 1 if weighted_sum >= 0 else -1
    return new_state

def sync_forward(sample, weights):
    weighted_sums = np.dot(weights, sample)
    
    new_sample = np.where(weighted_sums >= 0, 1, -1)
    
    return new_sample

def visualize_patterns(file_path, size):
    """
    Visualize all patterns in a dataset.

    Parameters:
    - file_path: Path to the dataset CSV file.
    - size: Tuple indicating the dimensions (rows, columns) to reshape the patterns for visualization.
    """
    # Load patterns from the file
    patterns = load_patterns(file_path)

    # Determine the number of patterns
    num_patterns = patterns.shape[0]

    # Create a grid for plotting
    rows = int(np.ceil(np.sqrt(num_patterns)))  # Number of rows in the grid
    cols = int(np.ceil(num_patterns / rows))   # Number of columns in the grid
    plt.figure(figsize=(cols * 3, rows * 3))   # Adjust the figure size for better visibility

    # Plot each pattern
    for i, pattern in enumerate(patterns):
        # Reshape the pattern to the specified size
        pattern_matrix = np.reshape(pattern, size)
        
        # Add a subplot to the grid
        plt.subplot(rows, cols, i + 1)
        plt.imshow(pattern_matrix, cmap='coolwarm', interpolation='nearest')
        plt.title(f"Pattern {i+1}")
        plt.axis('off')  # Turn off axes for better visualization

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_iteration_animation(states, size, step=50):

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
def evaluate_recovery(file_paths, sizes, noise_trials=10, noise_level=0.01):
    results = []

    for file_path, size in zip(file_paths, sizes):
        print(f"\nEvaluating dataset: {os.path.basename(file_path)}")

        patterns = load_patterns(file_path)
        num_patterns = patterns.shape[0]
        
        weights = train_hopfield(patterns, 'heb')
        stable_count = sum(is_stable(pattern, weights) for pattern in patterns)
        print(f"Total patterns: {num_patterns}")
        print(f"Stable patterns: {stable_count}")
        

        successful_recoveries = 0
        total_trials = noise_trials * len(patterns)
        
        for pattern in patterns:
            for _ in range(noise_trials):
                noisy_sample = generate_sample(pattern, flip_probability=noise_level)
                recovered_state = async_update(weights, noisy_sample, patterns)[-1]
                if np.array_equal(recovered_state, pattern):
                    successful_recoveries += 1
        
        recovery_rate = successful_recoveries / total_trials * 100
        print(f"Recovery success rate: {recovery_rate:.2f}%")
def main(file_path, size):
    patterns = load_patterns(file_path)
    
    #training
    weights = train_hopfield(patterns, 'heb') # heb/oja

    sample = generate_sample(patterns[2])

    states = async_update(weights, sample, patterns)

    # Plot all the states
    plot_iteration_animation(states, size)
def oscylacja():

    patterns = np.array([
        [1, -1],  
        [-1, 1]   
    ])
    
    weights = train_hopfield(patterns,'heb')

    x = np.array([-1, -1])  
    
    current_state = x
    states = [current_state]

    for _ in range(5): 
        new_state = sync_forward(current_state, weights)
        states.append(new_state)
        
        if np.array_equal(new_state, current_state): 
            break
        
        current_state = new_state

    for i, state in enumerate(states):
        print(f"Iteracja {i}: {state}")
def all_stable_test():

    patterns = np.array([
    [ 1, -1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1, -1],
    [-1, -1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1],
    [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
    [ 1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1, -1, -1,  1],
    [-1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1],
    [ 1, -1, -1,  1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1],
    [ 1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1],
    [ 1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1,  1],
    [ 1,  1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1],
    [-1, -1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1]
])
    weights = train_hopfield(patterns)
    for pattern in patterns:
        print(is_stable(pattern,weights))
    print(weights)

def generate_patterns(N, num_patterns):

    patterns = []
    
    for _ in range(num_patterns):
        pattern = np.random.choice([1, -1], size=N*N)
        patterns.append(pattern)
        
    patterns = np.array(patterns)
    return patterns
def generate_stable_patterns(N, y):

    while True:
        patterns = generate_patterns(N, y) 
        weights = train_hopfield(patterns) 
        
        stable = True
        for pattern in patterns:
            if not is_stable(pattern, weights):
                stable = False
                break
        
        if stable:
            print(f"All {y} patterns are stable.")
            print("Patterns:")
            for pattern in patterns:
                print(pattern)  # Print each pattern as an N x N grid
            y += 1
            

        # Increase the number of patterns to generate next time
def random_sample(file_path, size):
    patterns = load_patterns(file_path)
    
    #training
    weights = train_hopfield(patterns, 'heb') # heb/oja

    sample = np.random.choice([-1, 1], size=len(patterns[0]))

    states = async_update(weights, sample, patterns)

    # Plot all the states
    plot_iteration_animation(states, size)

def load_bmp_to_vector(filepath):
    img1 = Image.open(filepath + "/kot1.BMP")
    img2 = Image.open(filepath + "/kot2.BMP")
    
    # Convert the images to grayscale (L mode)
    grayscale_img1 = img1.convert("L")
    grayscale_img2 = img2.convert("L")

    # Convert the grayscale images to binary (-1, 1)
    binary_array1 = np.where(np.array(grayscale_img1) > 127, 1, -1)
    binary_array2 = np.where(np.array(grayscale_img2) > 127, 1, -1)

    # Flatten the binary arrays into 1D vectors
    vector1 = binary_array1.flatten()
    vector2 = binary_array2.flatten()
    
    # Combine vectors into a single 2D array
    patterns = np.array([vector1, vector2])
    
    # Train Hopfield network with the binary image data
    weights = train_hopfield(patterns, 'heb')  
    
    # Generate a noisy version of one pattern
    noisy_sample = np.random.choice([1, -1], size=grayscale_img1.size[1]*grayscale_img1.size[0])
    
    # Recover the noisy pattern using async update
    states = async_update(weights, noisy_sample, patterns,timeout=1000)
    
    # Plot the states during the recovery process
    plot_iteration_animation(states, grayscale_img1.size)

    
if __name__ == "__main__":
    path = "S:\SN\proj 2\Hopfield_Network\klastrowanie";

    file_paths = [
    path + '/animals-14x9.csv',
    path + '/large-25x25.csv',
    path + '/large-25x25.plus.csv',
    path + '/large-25x50.csv',
    path + '/letters-14x20.csv',
    path + '/letters-abc-8x12.csv',
    path + '/OCRA-12x30-cut.csv',
    path + '/small-7x7.csv'
    ]

    sizes = [
    (14, 9),
    (25, 25),
    (25, 25),
    (25, 50),
    (14, 20),
    (8, 12),
    (12, 30),
    (7, 7)
    ]
    pack = zip(file_paths, sizes)
    #load_bmp_to_vector(path)
    #evaluate_recovery(file_paths, sizes, noise_trials=10, noise_level=0.1)
    random_sample(path +'/large-25x25.csv', (25, 25))
    #main(path +'/large-25x50.csv', (25, 50)) #main(os.getcwd() +'/klastrowanie/large-25x25.csv', (25, 25))
    #all_stable_test()    
    #generate_stable_patterns(5,13)
    #oscylacja()
    #for i in range(len(sizes)):
    #    visualize_patterns(file_paths[i], sizes[i])



