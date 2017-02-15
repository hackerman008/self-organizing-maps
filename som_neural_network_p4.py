"""**************Self Organizing Maps for Dimensionality Reduction******************"""

import numpy as np
import pandas as pd
from sklearn import datasets
import copy
from scipy.spatial.distance import euclidean 
from scipy.spatial.distance import minkowski
import sklearn.metrics.pairwise as pair_wise
from ast import literal_eval
from itertools import groupby
from operator import itemgetter

class SOM(object):
    """initialize the newural network parameters
    number of neurons in the map = m*n
    numner of dimernsions of the input vector = dim
    dimensions of the weight vector = dim
    default learning rate = 0.3
    default no'of iteration = 100"""
    def __init__(self, m, n, dim, iterations=None, learning_rate=None, seed=None, model_type=None, initialization_type=None):
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
            rand_generator = np.random.RandomState(1) #default seed value for random generator
        else:
            rand_generator = np.random.RandomState(seed)
        """check for model type"""    
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
        self.__current_radius = 0   # variable to store radius after each iteration
        """intializing weight vectors for each node"""
        if initialization_type == None:
            print "no initialzation type given taking random initialization"
            self.__initialization_type = 'random'        
            self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        elif initialization_type == 'random':
            print "initialization weights as random"
            self.__initialization_type = 'random'          
            self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        elif initialization_type == 'other':
             print "initializing weights as node_index/no_of_nodes"
             self.__initialization_type = 'other'                    
             self.__number_of_nodes = self.__m*self.__n #total no'of nodes
             self.temp_list=[]        
             self.temp_list = [np.full(self.__dim,(float(i)/self.__number_of_nodes)) for i in range(self.__number_of_nodes)]        
             self.original_map_weight_vectors = np.reshape(self.temp_list,(self.__m,self.__n,dim))  
        else:
            print "invalid initialzation type given taking random initialization"
            self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)

                
        self.map_with_weight_vectors = copy.deepcopy(self.original_map_weight_vectors)
        #print "nodes afeter reshaping\n", self.original_map_weight_vectors       

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
        print("weight initialization type = ",self.__initialization_type)

    def find_bmu(self, input_vector):
        """find the best matching unit and return the index of the 
            node and the weights of the node"""
        
        index=[]
        value=[]
        for i in range(self.__m):
            array_of_distances = pair_wise.pairwise_distances(input_vector.reshape(1L,input_vector.shape[0]),self.map_with_weight_vectors[i])
            #print "array_of_distances=",array_of_distances            
            temp_index = np.argwhere(array_of_distances==np.min(array_of_distances))            
            #print "temp_index=",temp_index[0]            
            temp_index[0][0]=i   
            #print "changed_index=",temp_index[0]
            
            index.append(temp_index[0])
            temp_value = np.min(array_of_distances)
            #print "temp_value=",temp_value
            value.append(temp_value)
        
        return_index=index[np.argmin(value)]
        return_index=list(return_index)
        #print "return index=",return_index
        #print "weight vector=",self.map_with_weight_vectors[return_index[0]][return_index[1]]
        return self.map_with_weight_vectors[return_index[0]][return_index[1]], return_index

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
        old_quantization_error1 = 9999.0
        old_quantization_error2 = 9999.0
        
        while t < self.__iterations:
            print("*********************************************************\n*******************************************")
            print("timestep =",t)
            self.__cur_learning_rate = self.learning_rate(t)
            print("learning rate=", self.__cur_learning_rate)      
            self.__current_radius = self.find_radius_of_the_neighborhood(t)
            print("radius of the neighborhood", self.__current_radius)
            #shuffle the data and loop over the shuffled data
            np.random.shuffle(shuffled_data)            
            for i in range(len(shuffled_data)):        
                #pick a random sample
                #sample_index = np.random.randint(len(shuffled_data))
                self.bmu,self.bmu_index = self.find_bmu(shuffled_data[i]) #returns index of bmu node
                ##print "bmu weights=",self.bmu
                #print("bmu index=",self.bmu_index)
                self.update_weights(self.bmu_index,shuffled_data[i],t)
                
            """Reduction in Quantization error should not go below 0.01 and
               radius of the neighborhood should not go below 1 ever."""
            quantization_error = self.quantization_error(shuffled_data)
            print("Quantization error ", quantization_error)
            
            if((max(abs(old_quantization_error1 - quantization_error), abs(old_quantization_error2 - quantization_error))*100.0/old_quantization_error1) < 0.1):
                print "Quantization improvement went below 0.01"
                break
#            elif self.__current_radius <= 1:
#                print "radius of th eneighborhood went below 1"
#                break   
            t += 1
            old_quantization_error1 = old_quantization_error2
            old_quantization_error2 = quantization_error
        return 0
        
    def fine_tuning(self, data, ft_learning_rate=0.1, ft_radius=1.0, ft_iterations=10):
        """train the neural network"""
        
        shuffled_data = copy.deepcopy(data)
        t = 0
        self.__cur_learning_rate = ft_learning_rate
        self.__current_radius = ft_radius
        print("radius of the neighborhood", self.__current_radius)
        
        while t < ft_iterations:
            print("Fine tuning timestep =",t)
            #shuffle the data and loop over the shuffled data
            np.random.shuffle(shuffled_data)            
            for i in range(len(shuffled_data)):
                self.bmu,self.bmu_index = self.find_bmu(shuffled_data[i]) #returns index of bmu node
                ##print "bmu weights=",self.bmu
                #print("bmu index=",self.bmu_index)
                self.update_weights(self.bmu_index,shuffled_data[i],t)
            t += 1
        
        quantization_error = self.quantization_error(shuffled_data)
        print("Quantization error after fine tuning ", quantization_error)
        
        return 0
    
    def quantization_error(self,data):
        """calculating thw quantization error ,the lower the error,better the model"""
        numerator = 0        
        for i in range(len(data)):
            weight_vector,index = self.find_bmu(data[i])
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
        tp_error = float(error)/data.shape[0]
        return tp_error

    def dataframe_with_bmu_for_each_input_vector(self,data,list_of_columns, flag):
        """creating dataframe with indicator number and its BestMatchingUnit"""
        """flag =0 return dataframe with input and their bmu
            flag =1 return list with input and their bmu"""       
        temp_list = []
        for i in range(data.shape[0]): #data.index.values
            weight, index = self.find_bmu(data[i,:])
           # print "index=",index,"for data=",i#,"and target=",digits.target[i]
            temp_list.append((index,i))   
        if flag == 0:
            df = pd.DataFrame(temp_list, columns=['node','column'])        
            for i,x in enumerate(list_of_columns):
                df.ix[i,'column'] = x   
            print "return a data frame"
            return df            
        elif flag == 1:
             print "return list of input vector and their BMU"
             return temp_list
        
        
    def dataframe_with_bmu_for_each_input_vector_and_distance(self, passed_dataframe):
        """creating a dataframe with bmu indicator and the distance with each indicator"""
        
        temp_dataframe = copy.deepcopy(passed_dataframe)
        print temp_dataframe
        new_ind = pd.DataFrame(columns=[['bmu_indicator1','indicator_1','bmu_indicator2','indicator_2','distance']])
        print "dataframe size=",temp_dataframe.shape[0]
        for i in range(temp_dataframe.shape[0]):
            print "done for %d indicator"%(i)
            for j in range(temp_dataframe.shape[0]):
                #print "bmu1=",temp_dataframe.ix[i,0],"bmu2",temp_dataframe.ix[j,0]
                a = literal_eval(temp_dataframe.ix[i,'node'])
                b = literal_eval(temp_dataframe.ix[j,'node'])
                dist = minkowski(np.array(a),np.array(b),1)
                #print 'dist=',dist
                X = pd.DataFrame(np.array([[a, temp_dataframe.ix[i,'column'], 
                                               b, temp_dataframe.ix[j,'column'], dist]]),
                                              columns=['bmu_indicator1','indicator_1','bmu_indicator2','indicator_2','distance'])
                new_ind = new_ind.append(X, ignore_index=True)
        return new_ind
    
    def create_datasets(self, data, original_data, size, list_of_columns, list_of_node_and_input):
        #todo
        #sorted is necessary for grouping to work properly
        list_of_columns = original_data.columns.values.tolist()
        #original_data = original_data.values
        
        sorted_list_of_indicator_and_their_bmu = sorted(list_of_node_and_input)
        temp_list_to_store_clusters=[]       
        for k,v in groupby(sorted_list_of_indicator_and_their_bmu, key=itemgetter(0)):
            items = [x[1] for x in v]
            temp_list_to_store_clusters.append([k,items])
        
        print "grouping done"
        #list_of_columns_temp=list_of_columns#[2:(len(list_of_columns))]
        print "list_of_columns",len(list_of_columns)        
        df={} #dictionary to store 
        for i in range(size):
            df[i]=pd.DataFrame(columns=[list_of_columns])   #create empty dataframes and store it in dict            
        print "dataframes created"
        #looping over each cluster in the list and appending the required size of data in each dataframe
        var = 0        
        for i in range(len(temp_list_to_store_clusters)):
            print "for %d cluster"%(i)
            #starting_pos = 0
            #ending_pos = size

            var1 = 0 
            while var1<len(temp_list_to_store_clusters[i][1]):
                index = var%size
                df[index] = pd.concat((df[index], original_data.ix[[(temp_list_to_store_clusters[i][1][var1])]]), axis=0, ignore_index=True)
                #df[index] = df[index].append(pd.DataFrame(np.array([original_data[(temp_list_to_store_clusters[i][1][var1]),:]]), columns=list_of_columns), ignore_index=True)
                var = var + 1
                var1 = var1 + 1
            
        return df,temp_list_to_store_clusters
        
        
        
if __name__ == "__main__":
    
    #som = SOM(8, 8, 4,iterations = 500, seed=1,model_type=4)
    som = SOM(5, 5, 4,iterations = 10, seed=1,model_type='linear',initialization_type='random')
    som.print_variables()
    
    iris = datasets.load_iris()
    iris_data = iris.data
    from sklearn import preprocessing
    iris_data = preprocessing.scale(iris_data)
    som.train(iris_data)
    
    #find the best matching node for each data point
    for i in range(len(iris_data)):
        weight, index = som.best_matching_node(iris_data[i,:])
        print "index=",index,"for data=",i
        #,"and target=",digits.target[i]
        
    #print the topographic error ad quantization error
    print(som.topographic_error(iris_data))
    print(som.quantization_error(iris_data))
    
    #create different datasets , each data set contains samples from each cluster ,this dataset can be used for cross validation
    #and since the datasets will hav same statistical properties.
    list_of_columns=['one','two','three','four']
    list_of_node_and_input = som.dataframe_with_bmu_for_each_input_vector(iris_data, list_of_columns, flag=1)
    df, temp_list_to_store_clusters=som.create_datasets(iris_data, 4, list_of_columns, list_of_node_and_input)
  
