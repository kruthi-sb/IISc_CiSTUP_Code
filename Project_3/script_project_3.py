"""
Author:  Kruthi S B
Status:  Complete
Created:  23-03-2024
Completed: 24-03-2024
Description: This script is used to calculate the probability of each alternative in a multinomial choice setting using the logistic function, given a set of parameters and independent variables.
It also visualizes the probabilities using various plots.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

""" 
calculate_probabilities : function
Calculates the probability of each alternative in a multinomial choice setting using the logistic function.

parameters : dict
    Dictionary of parameters for the utility functions
data : dict
    Dictionary of independent variables
utility_values : list
    List of utility values for each alternative

Returns:
-------
probabilities : dict
    Dictionary of probabilities for each alternative
"""
def calculate_probabilities(parameters, data, utility_values):
    
    # error handling
    # check if the dimensions of the parameters and data match
    if len(parameters) != len(data): 
        raise ValueError("Mismatched dimensions between parameters and data points")

    # check if the no. of data points are the same for all independent variables
    num_data_points = len(data[list(data.keys())[0]])
    for key in data.keys():
        if len(data[key]) != num_data_points:
            raise ValueError("Mismatched dimensions between data points")
    
    # assuming the alternatives are appended to the dictionary at the end
    # get the number of alternatives from the utility values
    num_alternatives = len(utility_values)

    # get alternatives - the last num_alternatives keys and values
    alternatives_keys = list(data.keys())[-num_alternatives:]
    values = [data[key] for key in alternatives_keys]
    alternatives = dict(zip(alternatives_keys, values)) 

    # initialize the probabilities dictionary
    probabilities = {alternative: [] for alternative in alternatives_keys}

    # for each data point, calculate the sum of exponentials of the utility values multiplied by the alternative values
    for i in range(num_data_points):
        sum_exp = 0 # initialize the sum of exponentials
        for j in range(num_alternatives):
            sum_exp += np.exp(utility_values[j][i] * values[j][i]) # calculate the sum of exponentials

        # calculate the probabilities for each alternative
        for j in range(num_alternatives):
            numerator = np.exp(utility_values[j][i] * values[j][i]) # calculate the numerator
            probabilities[alternatives_keys[j]].append(numerator / sum_exp) # calculate the probability and append to the dictionary

    return probabilities

# test the function

# define the parameters 
parameters = {"b_01": 0.1,
              "b_1": -0.5,
              "b_2": -0.4,
              "b_02": 1,
              "b_03": 0,
              "b_s1_13": 0.33,
              "b_s1_23":0.58}

# define the data
# independent variables 7, av = 3
data = {
    'X1': [2,1,3,4,2,1,8,7,3,2],
    'X2': [8,7,4,1,4,7,2,2,3,1],
    'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'S1': [3,8,4,7,1,6,5,9,2,3],
    'AV1' : [1,1,1,1,1,0,0,1,1,0],
    'AV2': [1,1,1,0,0,1,1,1,0,1],
    'AV3': [1,1,0,0,1,1,1,1,1,1]
}

# deterministic utilities 3
# generate deterministic utilities for each alternative
# list of functions
utilities = [
    lambda parameters, data: parameters['b_01'] + np.multiply(parameters['b_1'], data['X1']) + np.multiply(parameters['b_s1_13'], data['S1']),
    lambda parameters, data: parameters['b_02'] + np.multiply(parameters['b_2'], data['X2']) + np.multiply(parameters['b_s1_23'], data['S1']),
    lambda parameters, data: parameters['b_03'] + np.multiply(parameters['b_1'], data['Sero']) + np.multiply(parameters['b_2'], data['Sero'])
]

# compute utility function
"""
calculate_utilities : function
Calculates the utility values for each alternative.

parameters : dict
    Dictionary of parameters for the utility functions
data : dict
    Dictionary of independent variables
utilities : list
    List of utility functions
    
Returns:
-------
utility_values : list
    List of utility values for each alternative
"""
def calculate_utilities(parameters, data, utilities):
    # check if the dimensions of the parameters and data match
    if len(parameters) != len(data):
        raise ValueError("Mismatched dimensions between parameters and data points")

    # check if the no. of data points are the same for all independent variables
    num_data_points = len(data[list(data.keys())[0]])
    for key in data.keys():
        if len(data[key]) != num_data_points:
            raise ValueError("Mismatched dimensions between data points")
        
    # get the independent variables as numpy arrays
    X1 = np.array(data['X1'])
    X2 = np.array(data['X2'])
    Sero = np.array(data['Sero'])
    S1 = np.array(data['S1'])

    # make a dictionary of the independent variables
    data = {'X1': X1, 'X2': X2, 'Sero': Sero, 'S1': S1}

    # initialize the utilities list
    # list of numpy arrays
    utility_values = []

    # calculate the utilities for each alternative
    for utility in utilities:
        utility_values.append(utility(parameters, data))

    return utility_values

# Visualizations
def bar_plot_datapoints(probabilities, data_point_index = 0):
    # Extract probabilities for the selected data point
    probabilities_data_point = {key: probabilities[key][data_point_index] for key in probabilities.keys()}

    # Create a bar plot
    plt.bar(probabilities_data_point.keys(), probabilities_data_point.values())
    plt.title('Probabilities of Alternatives for Data Point {}'.format(data_point_index + 1))
    plt.xlabel('Alternative')
    plt.ylabel('Probability')
    plt.show()
    
def line_plot_probabilities(probabilities):
    # Create a line plot for the probabilities of each alternative across data points
    for alternative in probabilities.keys():
        plt.plot(range(len(data['X1'])), probabilities[alternative], label=alternative)

    plt.title('Probabilities of Alternatives Across Data Points')
    plt.xlabel('Data Point Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_graph():
    # Generate utility values along the x-axis
    x_values = np.linspace(-5, 5, 100)

    # Calculate probabilities using the logistic function
    probabilities = sigmoid(x_values)

    # Plot the sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, probabilities, label='Sigmoid Function', color='blue')
    plt.title('Sigmoid Function for Multinomial Choice Model')
    plt.xlabel('Utility of Alternative')
    plt.ylabel('Probability of Choosing Alternative')
    plt.grid(True)
    plt.legend()
    plt.show()

def scatter_plot_utility_probability(utility_values, probabilities):
    # Extract utility values and probabilities for AV1 alternative
    utility_av1 = utility_values[0]  # Assuming AV1 is the first alternative
    prob_av1 = probabilities['AV1']

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(utility_av1, prob_av1, color='b', alpha=0.5)
    plt.title('Relationship between Utility and Probability of AV1 Alternative')
    plt.xlabel('Utility of AV1 Alternative')
    plt.ylabel('Probability of Choosing AV1 Alternative')
    plt.grid(True)
    plt.show()

# main script
if __name__ == "__main__":
    # calculate the utility values
    utility_values = calculate_utilities(parameters, data, utilities)

    # calculate the probabilities
    probabilities = calculate_probabilities(parameters, data, utility_values)

    # print the probabilities
    print("Probabilities:\n", probabilities,"\n")

    # check the sum of probabilities from each alternative for each data point
    print("Verification: Sum of probabilities for each data point:")
    for i in range(len(data['X1'])):
        sum_prob = 0
        for key in probabilities.keys():
            sum_prob += probabilities[key][i]
        print(sum_prob)

    # save the probabilities in a text file
    with open('probabilities.txt', 'w') as file:
        for key in probabilities.keys():
            file.write(f"{key}: {probabilities[key]}\n")
    file.close()

    # Visualizations
    # bar plot for probabilities of alternatives for a specific data point
    bar_plot_datapoints(probabilities, data_point_index = 0)

    # line plot for probabilities of alternatives across data points
    line_plot_probabilities(probabilities)

    # sigmoid function graph
    sigmoid_graph()

    # scatter plot for utility values and probabilities of AV1 alternative
    scatter_plot_utility_probability(utility_values, probabilities)

# end of script