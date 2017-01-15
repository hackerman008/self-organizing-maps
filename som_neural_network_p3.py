"""som prototype 2"""
"""**************Self Organizing Maps for Dimensionality Reduction******************"""
# http://ivape3.blogs.uv.es/2015/03/15/self-organizing-maps-the-kohonens-algorithm-explained/
import numpy as np
import pandas as pd
from sklearn import datasets
import copy
import os 
import sys
from scipy.spatial.distance import euclidean 
from scipy.spatial.distance import minkowski


class SOM(object):
    """initialize the newural network parameters
    number of neurons in the map = m*n
    numner of dimernsions of the input vector = dim
    dimensions of the weight vector = dim
    default learning rate = 0.3
    default no'of iteration = 100"""
    def __init__(self, m, n, dim, iterations=None, learning_rate=None, seed=None):
        self.__m = m
        self.__n = n
        self.__dim = dim
        if iterations is None:
            self.__iterations = 100
        else:
            self.__iterations = iterations
        if learning_rate == None:
            self.__learning_rate = 1
        else:
            self.__learning_rate = learning_rate
        if seed == None:
            rand_generator = np.random.RandomState(1)
        else:
            rand_generator = np.random.RandomState(seed)
        #int((max(5,4))/2.0)
        self.__map_radius = round(max(self.__m, self.__n)/2)
        #self.__lambda = self.__iterations/self.__map_radius
        self.__lambda = 1500
        self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        #self.map_with_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        self.map_with_weight_vectors = copy.deepcopy(self.original_map_weight_vectors)

    def print_variables(self):
        """printing variables"""
        print("m =", self.__m, "\tn =", self.__n)
        print("dimensions of the input vector = ", self.__dim)
        print("learning rate = ", self.__learning_rate)
        print("no of iterations =", self.__iterations)
        print("map radius = ", self.__map_radius)
        print("lambda = ", self.__lambda)
        print("map shape = ", self.map_with_weight_vectors.shape)

    '''
    def initialize_weights(self,m,n,dim):
        #should return an array of weights for each node
    '''

    def find_bmu(self, input_vector):
        """find the best matching unit
            and return the index of the 
            node and the weights of the
            node"""
        min_distance = np.inf
        index = [0, 0]
        for i in range(self.__m):
            for j in range(self.__n):
                # np.linalg.norm helps to calculate the  distance based on the norm
                distance = 0
                distance = euclidean(input_vector, self.map_with_weight_vectors[i][j])
                #print "distance for i=",i,"j=",j,"distance =",distance
                #print "weight of the node",self.map_with_weight_vectors[i][j]
                if distance < min_distance:
                    min_distance = distance
                    index = [i, j]
                else:
                    pass
        return self.map_with_weight_vectors[index[0]][index[1]],index

    def find_radius_of_the_neighborhood(self, time_value):
        """find the radius of the neighborhood for the current time value"""        
        radius_of_the_neighborhood = self.__map_radius*(np.exp(-(time_value/self.__lambda)))
        return radius_of_the_neighborhood
        
    def update_weights(self,bmu_index,input_vector,time_step):
        """update the weights of the BMU including the neighborhood nodes"""        
        for i in range(self.__m):
            for j in range(self.__n):
                #print "for node [%d,%d]"%(i,j)
                index=[i,j]
                print("for node=",index)
                a = self.learning_rate(time_step)
                self.map_with_weight_vectors[i][j] = self.map_with_weight_vectors[i][j] + (a * self.calculate_neighborhoood(bmu_index,np.array(index),time_step)*(input_vector - self.map_with_weight_vectors[i][j]))                         
                    
        return 0

    def learning_rate(self,time_step):
        """calculate the learning rate"""
        
        alpha = self.__learning_rate*(np.exp(-(time_step/self.__lambda)))        
        print("learning rate=",alpha)        
        #print "alpha=",alpha
        return alpha

    def calculate_neighborhoood(self,bmu_index,current_node_index,time_step):
        """calculate the distance between the bmu and the node"""
        
        distance = euclidean(bmu_index,current_node_index)
        radius_of_the_neighborhood = self.find_radius_of_the_neighborhood(time_step)
        theta = np.exp(-(distance**2/(2*(radius_of_the_neighborhood)**2)))
        print("neighborhood function value =",theta)
        print("radius of the neighborhood=",radius_of_the_neighborhood)        
        #print "theta=",theta
        return theta

    def train(self, data):
        """train the neural network"""
        #self.array_of_node_with_weights = self.initialize_weights(self.__m,self.__n,self.__dim)

        t = 0
        while t < self.__iterations:
            print("*********************************************************\n*******************************************")
            print("timestep =",t)
            #pick a random sample
            sample_index = np.random.randint(len(data))
            #print "for iteration ", t, " data = ", data[sample_index], "index = ", sample_index
            self.bmu,self.bmu_index = self.find_bmu(data[sample_index]) #returns index of bmu node
            #print "bmu weights=",self.bmu
            print("bmu index=",self.bmu_index)
            """learning rate should not go below 0.01 in first phase and
                radius of the neighborhood should not go below 0 ever."""
            if self.learning_rate(t) < 0.01:
                break
            elif self.find_radius_of_the_neighborhood(t) <= 0:
                break
            self.update_weights(self.bmu_index,data[sample_index],t)
            t += 1
        return 0
    
    def best_matching_node(self,data):
        """find the best matching unit for each passed data point"""
        dist = np.inf
        index = []
        for i in range(self.__m):
            for j in range(self.__n):
                temp_dist = 0
                temp_dist = euclidean(data,self.map_with_weight_vectors[i][j])
                if temp_dist < dist:
                    dist = temp_dist
                    index = [i, j]
                else:
                    pass
        return self.map_with_weight_vectors[index[0]][index[1]],index
        
    def quantization_error(self,data):
        """calculating thw quantization error ,the lower the error,better the model"""
        numerator = 0        
        for i in range(len(data)):
            weight_vector,index = self.best_matching_node(data[i])
            numerator = numerator + euclidean(data[i],weight_vector) #ord = 1 or 2
            
        error = numerator/data.shape[0]
        return error        

    def calculate_difference(self,input_vector):
        """calculate the distance of the input vector with each node and
            return the indices of the node with least distance."""
        list_for_storing_distance_with_each_node = []
        for i in range(self.__m):
            for j in range(self.__n):
                temp_distance = euclidean(input_vector, self.map_with_weight_vectors[i][j])                
                list_for_storing_distance_with_each_node.append((temp_distance,[i,j]))
                #print "temp distance with node [%d %d] is %f"%(i,j,temp_distance)
        
        sorted_list_for_storing_distance_with_each_node = sorted(list_for_storing_distance_with_each_node)
        return sorted_list_for_storing_distance_with_each_node[0][1], sorted_list_for_storing_distance_with_each_node[1][1] 
        
    def check_if_neighbors(self, index1, index2):
        """check if the two nodes are neighbors.
            If yes return 1 else 0."""
        if (index1[0] == index2[0]) or (index1[1] == index2[1]):
            val = 1
        else:
            val = 0
        return val
    
    def topographic_error(self,data):
        """calculate topographic error,if the second bmu is not the neighbor of the first then 1 else 0"""
        error = 0
        for i in range(data.shape[0]):
            #print "for input=",i
            #print "input vector=",data[i]
            index_of_bmu1, index_of_bmu2 = self.calculate_difference(data[i])
            #print "index 1=",index_of_bmu1,"index 2=",index_of_bmu2            
            val = self.check_if_neighbors(index_of_bmu1, index_of_bmu2)
            #print "val=",val            
            if val == 1:
                error = error + val
        tp_error = float(error)/data.shape[0]
        return tp_error



if __name__ == "__main__":
    
    som = SOM(25, 25, 64,iterations = 1500, seed=2)
    som.print_variables()
    
    iris = datasets.load_iris()
    iris_data = iris.data
    
    digits = datasets.load_digits()
    digits_data = digits.data
    from sklearn import preprocessing
    iris_data = preprocessing.scale(digits_data)
    som.train(digits_data)
    """
    for i in range(149):
        weight, index = som.best_matching_node(iris_data[i,:])
        print "index=",index,"for data=",i
        #,"and target=",digits.target[i]
    """ 
    print(som.topographic_error(digits_data))
    print(som.quantization_error(digits_data))
   
    #print "original map=\n",som.original_map_weight_vectors
    #print "map after training=\n",som.map_with_weight_vectors    
