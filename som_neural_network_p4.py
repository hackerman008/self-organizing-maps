"""som prototype 2"""
"""**************Self Organizing Maps for Dimensionality Reduction******************"""
# http://ivape3.blogs.uv.es/2015/03/15/self-organizing-maps-the-kohonens-algorithm-explained/
#https://notendur.hi.is//~benedikt/Courses/Mia_report2.pdf
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
    def __init__(self, m, n, dim, iterations=None, learning_rate=None, seed=None, model_type=None):
        if m <= 0:
            print "m cannot be less than 1, setting it to 10"
            m = 10
        if n <= 0:
            print "n cannot be less than 1, setting it to 10"
            n = 10
        self.__m = m
        self.__n = n
        self.__dim = dim
        if iterations is None:
            self.__iterations = 50
        else:
            self.__iterations = iterations
        if learning_rate == None:
            self.__learning_rate = 0.8
            self.__cur_learning_rate = 0.8
        else:
            self.__learning_rate = learning_rate
            self.__cur_learning_rate = learning_rate
        if seed == None:
            rand_generator = np.random.RandomState(1)
        else:
            rand_generator = np.random.RandomState(seed)
            
        if model_type == None:
            self.__model_type= 0    #defaul linear model
        elif model_type == 'linear':
            self.__model_type = 0
        elif model_type == 'exp':
            self.__model_type = 1
        else:
            print "invalid modeltype, defaulting to linear"
            self.__model_type = 0
        
        self.__map_radius = round(max(self.__m, self.__n)/2)
        self.__lambda = self.__iterations/self.__map_radius
        self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        self.map_with_weight_vectors = copy.deepcopy(self.original_map_weight_vectors)
        self.__current_radius = 0

    def print_variables(self):
        """printing variables"""
        print("model type=",self.__model_type)
        print("m =", self.__m, "\tn =", self.__n)
        print("dimensions of the input vector = ", self.__dim)
        print("learning rate = ", self.__learning_rate)
        print("no of iterations =", self.__iterations)
        print("map radius = ", self.__map_radius)
        print("lambda = ", self.__lambda)
        print("map shape = ", self.map_with_weight_vectors.shape)


    def find_bmu(self, input_vector):
        """find the best matching unit and return the index of the 
            node and the weights of the node"""
            
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
        return self.map_with_weight_vectors[index[0]][index[1]], index

    def find_radius_of_the_neighborhood(self, time_value):
        """find the radius of the neighborhood for the current time value"""
        
        if self.__model_type == 0:  #if the model is using linear decay
            radius_of_the_neighborhood = 0.25 * (self.__m + self.__n) * (1.0 - (float(time_value)/self.__iterations))
        elif self.__model_type == 1: #if the model is using exponential decay
            radius_of_the_neighborhood = self.__map_radius*(np.exp(-(float(time_value)/self.__lambda)))
        return radius_of_the_neighborhood
        
    def update_weights(self,bmu_index,input_vector, time_step):
        """update the weights of the BMU including the neighborhood nodes"""   
        
        for i in range(self.__m):
            for j in range(self.__n):
                #print "for node [%d,%d]"%(i,j)
                index=[i,j]
                #print("for node=",index)
                self.map_with_weight_vectors[i][j] = self.map_with_weight_vectors[i][j] + (self.__cur_learning_rate * self.calculate_neighborhoood(bmu_index,np.array(index), time_step) * (input_vector - self.map_with_weight_vectors[i][j]))
        return 0

    def learning_rate(self,time_step):
        """calculate the learning rate"""
        
        if self.__model_type == 0:  #if the model is using linear decay
            alpha = 0.8*(1.0 - (float(time_step)/(6*self.__iterations)))
        elif self.__model_type == 1: #if model is using exponential decay       
            alpha = self.__learning_rate*(np.exp(-(float(time_step)/self.__lambda)))
        alpha = max(alpha, 0.01)
        #print "alpha=",alpha
        return alpha

    def calculate_neighborhoood(self,bmu_index,current_node_index, time_step):
        """calculate the distance between the bmu and the node"""
        
        distance = euclidean(bmu_index,current_node_index)
        radius_of_the_neighborhood = self.__current_radius
        if self.__model_type == 0:  #if the model is using linear decay
            theta = np.exp(-(float(distance)/(2*(radius_of_the_neighborhood)**2)))
        elif self.__model_type == 1: #if model is using exponential decay
            theta = np.exp(-(distance**2/(2*(radius_of_the_neighborhood)**2)))
        #print("neighborhood function value =",theta)
        #print("radius of the neighborhood=",radius_of_the_neighborhood)        
        #print "theta=",theta
        return theta

    def train(self, data):
        """train the neural network"""
        
        shuffled_data = copy.deepcopy(data)
        t = 0
        old_quantization_error = 9999.0
        
        while t < self.__iterations:
            print("*********************************************************\n*******************************************")
            print("timestep =",t)
            self.__cur_learning_rate = self.learning_rate(t)
            print("learning rate=", self.__cur_learning_rate)      
            self.__current_radius = self.find_radius_of_the_neighborhood(t)
            print("radius of the neighborhood")
            #shuffle the data and loop over the shuffled data
            np.random.shuffle(shuffled_data)            
            for i in range(len(shuffled_data)):        
                #pick a random sample
                #sample_index = np.random.randint(len(shuffled_data))
                self.bmu,self.bmu_index = self.find_bmu(shuffled_data[i]) #returns index of bmu node
                ##print "bmu weights=",self.bmu
                #print("bmu index=",self.bmu_index)
                self.update_weights(self.bmu_index,data[i],t)
                
            """Reduction in Quantization error should not go below 0.01 and
               radius of the neighborhood should not go below 1 ever."""
            quantization_error = self.quantization_error(shuffled_data)
            
            if (old_quantization_error - quantization_error) < 0.05:
                print "Quantization improvement went below 0.01"
                break
#            elif self.__current_radius <= 1:
#                print "radius of th eneighborhood went below 1"
#                break   
            t += 1
            old_quantization_error = quantization_error
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
            
        error = float(numerator)/data.shape[0]
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
        """topographic error if distance is 1 error 0,
            if distance is 2 error 0.5 , 
            if more than 2 error is 1."""
        distance_between_index_and_index2 = minkowski(index1, index2, 1)
        if distance_between_index_and_index2 == 1:
            val = 0
        elif distance_between_index_and_index2 == 2:
            val = 0.5
        elif distance_between_index_and_index2 > 2:
            val = 1
        """   
        if (index1[0] == index2[0]) or (index1[1] == index2[1]):
            val = 1
        else:
            val = 0
        """
        return val
    
    def topographic_error(self,data):
        """calculate topographic error,if the second bmu is not the neighbor of the first then 1 else 0"""
        error = 0
        for i in range(data.shape[0]):
            #print "for input=",i
            #print "input vector=",data[i]
            index_of_bmu1, index_of_bmu2 = self.calculate_difference(data[i])
            #print "index 1=",index_of_bmu1,"index 2=",0.793333333333index_of_bmu2            
            val = self.check_if_neighbors(index_of_bmu1, index_of_bmu2)
            error = error + val
            #print "val=",val
            """
            if val == 1:
                error = error + val
            """
        tp_error = float(error)/data.shape[0]
        return tp_error

    def dataframe_with_bmu_for_each_input_vector(self,data,list_of_columns):
        b = []
        for i in range(data.shape[0]):
            weight, index = self.best_matching_node(data[i,:])
           # print "index=",index,"for data=",i#,"and target=",digits.target[i]
            b.append((index,i))
        df = pd.DataFrame(b, columns=['node','column'])
        for i in range(len(df)):
            df.ix[i,'column'] = list_of_columns[i+2]
        return df

    def cluster(self,data):
        self.train(data)
        temp_list = [] 
        for i in range(data.shape[0]):
            weight, index = self.best_matching_node(data[i,:])
            print "index=",index,"for data=",i#,"and target=",digits.target[i]
            temp_list.append((index,i))        
        print "done"      
        return temp_list



if __name__ == "__main__":
    
    som = SOM(8, 8, 4,iterations = 500, seed=1,model_type=4)
    som.print_variables()
    
    iris = datasets.load_iris()
    iris_data = iris.data
    
    digits = datasets.load_digits()
    digits_data = digits.data
    from sklearn import preprocessing
    iris_data = preprocessing.scale(iris_data)
    som.train(iris_data)
    """
    for i in range(149):
        weight, index = som.best_matching_node(iris_data[i,:])
        print "index=",index,"for data=",i
        #,"and target=",digits.target[i]
    """ 
    print(som.topographic_error(iris_data))
    print(som.quantization_error(iris_data))
    #som.train(iris_data)
    #print "original map=\n",som.original_map_weight_vectors
    #print "map after training=\n",som.map_with_weight_vectors    