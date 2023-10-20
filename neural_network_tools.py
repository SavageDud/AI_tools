# -*- coding: utf-8 -*-

import math
import numpy as np
import copy
import random

def linear_function(x):
    return x

def binary_step(x):
    if(x < 0):
        return 0
    return 1

def sigmoid(x):
    return 1 / (1 + math.pow(2.71 , -x))

def ReLU(x):
    return max(0 , x)


activation_funcs = {
    "lin_" : linear_function,
    "sigm" : sigmoid,
    "bin_step" : binary_step,
    "ReLU" : ReLU,
    }



def RandInt(lenght,scale):
    array = []
    for iteration in range(0 , lenght):
        array.append(random.randint(-scale, scale))
    return array


def Rand01Float(lenght , step):
    array = []
    exponent = 1 * step
    for iteration in range(0 , lenght):
        array.append(random.randrange(- exponent , exponent) / exponent )
    return array

print(Rand01Float(5 , 1000))

randomizing_methodes = {
    "RandInt" : RandInt,
    "Rand01Float" : Rand01Float,
}






class node:
    def __init__(self, function_type , weights , biase):
        self.func_type = function_type
        self.params = np.array((weights + [biase]))
        self.value = 0
    def compute_node(self , prev_matrix):
        return activation_funcs[self.func_type]((self.params * prev_matrix).sum())
    
    
    
    
    
class layer:
    def __init__(self , numberof_nodes , type_ , length_of_previous_layer):
        self.nodes = []
        for node_index in range(0 , numberof_nodes):
            self.nodes.append(node(type_[node_index] , [1] * length_of_previous_layer , 0))
            
    def compute_layer(self , prev_matrix):
        output_ = []
        for node_index in range(0 , len(self.nodes)):
            output_.append(self.nodes[node_index].compute_node(prev_matrix))
        return np.array(output_ + [1])
            
    
    
    




#this model will be 2 dimentional
class model:
    def __init__(self,input_size):
        
        self.layers = []
        self.input_lenght = input_size
        self.layer_size = [input_size]
        
    def load_save(self):
        return
    def save_state(self):
        return
    def create_layer(self, layer_size , types):
        
        new_layer = layer(layer_size , types , self.layer_size[-1])
        self.layers.append(new_layer)
        self.layer_size.append(layer_size)
    
    def compute_model(self, input_):
        
        layer_input = input_
        if(layer_input.shape != (self.input_lenght,)):
            print("shape of input must be the same as input shape")
            return
        
        layer_input = np.append(layer_input, [1])
        for layer_index in range(0 , len(self.layers)):
            layer_input  = self.layers[layer_index].compute_layer(layer_input)
        
        return  np.delete(layer_input , [layer_input.shape[0]- 1])
    
    #this will print the structure of the model
    def graph_model(self):
        structure = [self.input_lenght * ['input']]
        for layer_index in range(0 , len(self.layers)):
            layer_data = []
            for node_index in range(0 , len(self.layers[layer_index].nodes)):
                layer_data.append(self.layers[layer_index].nodes[node_index].func_type)
            structure.append(layer_data)
            
        #display stuff
        for layer in structure:
            print(layer)
    
        return structure
    
    
    
    #very memory stupid since we simulate one at the time
    def stepvariance(self , stepsize):
        #here will create a array of all posible one value variation
        #for example if we have a model with 9 total param
        #we will create 9 variation where we change one value positivly
        #and another 9 where we change them negativily
        all_model_variation = []
        for layer_index in range(0,len(self.layers)):
            for node_index in range(0 , len(self.layers[layer_index].nodes)):
                for param_index in range(0 , self.layers[layer_index].nodes[node_index].params.shape[0]):
                    negative_copy = copy.deepcopy(self)
                    positive_copy = copy.deepcopy(self)
                    positive_copy.layers[layer_index].nodes[node_index].params[param_index] += stepsize
                    negative_copy.layers[layer_index].nodes[node_index].params[param_index] -= stepsize
                    all_model_variation.append(positive_copy)
                    all_model_variation.append(negative_copy)
        
        
        return all_model_variation

    def len_step_variance(self):
        sum_ = 0
        for layer_index in range(0 , len(self.layers)):
            sum_ += len(self.layers[layer_index].nodes) * self.layers[layer_index].nodes[0].params.shape[0]
        return sum_ * 2
    

    def write_params(self):
        params = []
        for layer_index in range(0 , len(self.layers)):
            for node_index in range(0 , len(self.layers[layer_index].nodes)):
                params.append(self.layers[layer_index].nodes[node_index].params.tolist())
        return params
    
    def create_random_clone(self , state,scale ,methode):
        new_params = []
        for layer in state:
            new_params.append(randomizing_methodes[methode](len(state)))
    
    #I am here
    def adopte_random_state(self , save , methode,scale):
        
        return
    
    
    def num_to_stepvariance(self):
        return
    
    
    def overwrite_current_state(self, new_state):
        return
    
    
    def randomize_params(self , methode):
        current_state = np.array(self.write_params())
        random_state = np.array(self.create_random_clone(current_state,1000, "Rand01Float"))
        
        
        return




# agent = model(3)
# agent.create_layer(4 , ["ReLU"] * 5)
# agent.create_layer(4,  ["sigm"] * 5)

# print(agent.compute_model(np.array([1,4,2])))
# print(agent.write_params())

