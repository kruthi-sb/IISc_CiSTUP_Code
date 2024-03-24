# requirements
""" A function
- calculate probabilities of each alternative
- Multinomial Choice Setting
- use Logistic (sigmoid) function
- given: parameters(betas) and independent variables (X1, X2, Sero and S1)

- shld handle any no. of alternatives (y) and ind. var.
"""

""" Details
calculate_probabilities
- inputs: parameters (dict), data (dict, ind. var.), utilities (list of functions)

- output: dict: keys - alternatives, values - [list] of calculated probabilities for each data point.

- save in txt format

- error handling: mismatched dimensions between parameters and data points.
"""
import numpy as np

# 7
parameters = {"b_01": 0.1,
              "b_1": -0.5,
              "b_2": -0.4,
              "b_02": 1,
              "b_03": 0,
              "b_s1_13": 0.33,
              "b_s1_23":0.58}

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
# list of fucntions
utilities = [
    lambda parameters, data: parameters['b_01'] + np.multiply(parameters['b_1'], data['X1']) + np.multiply(parameters['b_s1_13'], data['S1']),
    lambda parameters, data: parameters['b_02'] + np.multiply(parameters['b_2'], data['X2']) + np.multiply(parameters['b_s1_23'], data['S1']),
    lambda parameters, data: parameters['b_03'] + np.multiply(parameters['b_1'], data['Sero']) + np.multiply(parameters['b_2'], data['Sero'])
]

# compute utility fucntions
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

    data = {'X1': X1, 'X2': X2, 'Sero': Sero, 'S1': S1}

    # initialize the utilities list
    # list of numpy arrays
    utility_values = []

    # calculate the utilities for each alternative
    for utility in utilities:
        utility_values.append(utility(parameters, data))

    return utility_values

# print the utility values
utility_values = calculate_utilities(parameters, data, utilities)
print(utility_values)


import numpy as np

# assumption = 3 alternatives only, so, 3 utilities
def calculate_probabilities(parameters, data, utility_values):
    
    # check if the dimensions of the parameters and data match
    if len(parameters) != len(data):
        raise ValueError("Mismatched dimensions between parameters and data points")

    # check if the no. of data points are the same for all independent variables
    num_data_points = len(data[list(data.keys())[0]])
    for key in data.keys():
        if len(data[key]) != num_data_points:
            raise ValueError("Mismatched dimensions between data points")
    
    # assuming the alternatives are appended to the dictionary at the end
    num_alternatives = len(utilities)

    # get alternatives - the last num_alternatives keys and values
    alternatives_keys = list(data.keys())[-num_alternatives:]
    values = [data[key] for key in alternatives_keys]
    alternatives = dict(zip(alternatives_keys, values)) 
    print(alternatives)

    # initialize the probabilities dictionary
    probabilities = {alternative: [] for alternative in alternatives_keys}
    print(probabilities)

    # calculate the sum of exponentials of all the utility values multiplied by the alternative specific parameter
    
    # for each data point, calculate the sum of exponentials of the utility values multiplied by the alternative values
    for i in range(num_data_points):
        sum_exp = 0
        for j in range(num_alternatives):
            sum_exp += np.exp(utility_values[j][i] * values[j][i])

        # calculate the probabilities for each alternative
        for j in range(num_alternatives):
            numerator = np.exp(utility_values[j][i] * values[j][i])
            probabilities[alternatives_keys[j]].append(numerator / sum_exp)

    return probabilities

# calculate the probabilities
probabilities = calculate_probabilities(parameters, data, utility_values)
print(probabilities)

# check the sum of probabilities from each alternative for each data point
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


import matplotlib.pyplot as plt

# Select a specific data point (e.g., the first one)
data_point_index = 0

# Extract probabilities for the selected data point
probabilities_data_point = {key: probabilities[key][data_point_index] for key in probabilities.keys()}

# Create a bar plot
plt.bar(probabilities_data_point.keys(), probabilities_data_point.values())
plt.title('Probabilities of Alternatives for Data Point {}'.format(data_point_index + 1))
plt.xlabel('Alternative')
plt.ylabel('Probability')
plt.show()

# Create a line plot for each alternative
for alternative in probabilities.keys():
    plt.plot(range(len(data['X1'])), probabilities[alternative], label=alternative)

plt.title('Probabilities of Alternatives Across Data Points')
plt.xlabel('Data Point Index')
plt.ylabel('Probability')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the logistic function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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



probabilities = calculate_probabilities(parameters, data, utility_values)




import matplotlib.pyplot as plt

# Extract utility values and probabilities for AV1 alternative
utility_av1 = utility_values[1]  # Assuming AV1 is the first alternative
print(utility_av1)
print(type(probabilities))
prob_av1 = probabilities['AV2']
print(prob_av1)

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(utility_av1, prob_av1, marker='o', linestyle='-', color='b')
plt.title('Relationship between Utility and Probability of AV1 Alternative')
plt.xlabel('Utility of AV1 Alternative')
plt.ylabel('Probability of Choosing AV1 Alternative')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

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