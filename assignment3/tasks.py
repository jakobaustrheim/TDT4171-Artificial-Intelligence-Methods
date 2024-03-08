import numpy as np

evidence = [True, True]

def forward(list): #Defining a function that takes in a list consisting of given evidence
    evidence = list 

    probability = np.array([0.5, 0.5]) #A list which values represent the initial probability

    transition_matrix = np.array([np.array([0.7, 0.3]), np.array([0.3,0.7])]) #Dynamic model from task 1
    sensor_matrix = np.array([np.array([0.9, 0.2]), np.array([0.2, 0.9])]) #Observation model from task 1

    for i in range(len(evidence)):
        prediction = transition_matrix[0] * probability[0] + transition_matrix[1] * probability[1] #Probability of rain the next day given the previous probability
        if evidence[i]: #If the given evidence of umbrella is True 
            unormalized_probability = sensor_matrix[0] * prediction
            a = 1 / (unormalized_probability[0] + unormalized_probability[1]) #Normalizing factor a
            probability = a * unormalized_probability #The normalized probability
        else: #If the given evidence of umbrella is False 
            unormalized_probability = sensor_matrix[1] * prediction
            a = 1 / (unormalized_probability[0] + unormalized_probability[1]) #Normalizing factor a
            probability = a * unormalized_probability #The normalized probability
    return probability

print(forward(evidence))
    

