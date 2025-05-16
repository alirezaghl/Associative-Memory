import numpy as np
import copy 
import random
import matplotlib.pyplot as plt

def Initializer(numNeurons):
    """
    initializes the weight matrix and threshold
    """
    w = np.zeros((numNeurons, numNeurons))
    theta = np.zeros(numNeurons)
    return w, theta

def Learner(w, theta, patterns):
    """
    trains the Hopfield network using Hebbian learning
    """
    numNeurons = len(w)

    for pattern in patterns:
        assert all(val == 1 or val == -1 for val in pattern)
        for i in range(numNeurons):
            for j in range(numNeurons):
                if i != j:  
                    w[i][j] += pattern[i] * pattern[j]
    
    # Set diagonal weights to zero
    for i in range(numNeurons):
        w[i][i] = 0
    
    for i in range(numNeurons):
        theta[i] = 0 

    return w, theta 

def Inference(w, theta, initialState, MaxIter):
    numNeurons = len(initialState)
    currentState = np.copy(initialState) 

    for _ in range(MaxIter): 
        oldState = np.copy(currentState) 
        
        indices = list(range(numNeurons))
        random.shuffle(indices)
        
        for i in indices:
            net_input = 0 
            for j in range(numNeurons):
                net_input += w[i][j] * currentState[j]
            
            net_input += theta[i]

            if net_input > 0:
                currentState[i] = 1
            elif net_input < 0:
                currentState[i] = -1

        if np.array_equal(currentState, oldState):
            break
            
    return currentState

def Energy(w, theta, state):
    """
    calculates the energy of the Hopfield network for a given state.
    """
    numNeurons = len(state)
    energy = 0 

    for i in range(numNeurons):
        for j in range(numNeurons):
            energy -= 0.5 * w[i][j] * state[i] * state[j]
    
    for i in range(numNeurons):
        energy -= theta[i] * state[i] 
    
    return energy




def test_hopfield_network():
  
    pattern = np.array([
        1,  1,  1,  1,
        -1, -1, -1, 1,
        1,  1, -1, 1,
        -1, -1, -1, 1
    ])
    numNeurons = len(pattern)
    w, theta = Initializer(numNeurons)
    w, theta = Learner(w, theta, [pattern]) 
    corrupted_pattern = np.copy(pattern)
    num_bits_to_flip = numNeurons // 4
    flip_indices = random.sample(range(numNeurons), num_bits_to_flip)
    for index in flip_indices:
        corrupted_pattern[index] *= -1
    
   
    MaxIter = 100 
    recovered_pattern = Inference(w, theta, corrupted_pattern, MaxIter)
    
    
    side_length = int(np.sqrt(numNeurons))
    pattern_reshaped = pattern.reshape(side_length, side_length)
    corrupted_reshaped = corrupted_pattern.reshape(side_length, side_length)
    recovered_reshaped = recovered_pattern.reshape(side_length, side_length)
        
    fig, axes = plt.subplots(1, 3) 

    axes[0].imshow(pattern_reshaped, cmap='binary', vmin=-1, vmax=1)
    axes[0].set_title('Original Pattern')
    axes[0].axis('off') 

    axes[1].imshow(corrupted_reshaped, cmap='binary', vmin=-1, vmax=1)
    axes[1].set_title('Corrupted Pattern')
    axes[1].axis('off')

    axes[2].imshow(recovered_reshaped, cmap='binary', vmin=-1, vmax=1)
    axes[2].set_title('Recovered Pattern')
    axes[2].axis('off')  
    plt.show()

test_hopfield_network()