"""**************Self Organizing Maps for Dimensionality Reduction******************"""
- http://stats.stackexchange.com/questions/22774/dimensionality-reduction-using-self-organizing-map
-http://stats.stackexchange.com/questions/64659/using-self-organizing-maps-for-dimensionality-reduction?noredirect=1&lq=1
import numpy as np
from sklearn import datasets
import copy

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
            self.__learning_rate = 0.3
        else:
            self.__learning_rate = learning_rate
        if seed == None:
            rand_generator = np.random.RandomState(1)
        else:
            rand_generator = np.random.RandomState(seed)
        #int((max(5,4))/2.0)
        self.__map_radius = round(max(self.__m, self.__n)/2)
        self.__lambda = self.__iterations/self.__map_radius
    
        self.original_map_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        #self.map_with_weight_vectors = rand_generator.rand(self.__m, self.__n, self.__dim)
        self.map_with_weight_vectors = copy.deepcopy(self.original_map_weight_vectors)

    def print_variables(self):
        """printing variables"""
        print "m =", self.__m, "\tn =", self.__n
        print "dimensions of the input vector = ", self.__dim
        print "learning rate = ", self.__learning_rate
        print "no of iterations =", self.__iterations
        print "map radius = ", self.__map_radius
        print "lambda = ", self.__lambda
        print "map shape = ", self.map_with_weight_vectors.shape

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
                distance = np.linalg.norm(input_vector-(self.map_with_weight_vectors[i][j]), ord=2)
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
        
    def update_weights(self,bmu,data,time_step):
        """update the weights of the BMU including the neighborhood nodes"""        
        for i in range(self.__m):
            for j in range(self.__n):
                #print "for node [%d,%d]"%(i,j)
                self.map_with_weight_vectors[i][j] = self.map_with_weight_vectors[i][j] + (self.learning_rate(time_step) * self.distance_from_bmu(bmu,self.map_with_weight_vectors[i][j],time_step)*(data - self.map_with_weight_vectors[i][j]))                         
        return 0


    def learning_rate(self,time_step):
        """calculate the learning rate"""
        
        alpha = self.__learning_rate*(np.exp(-(time_step/self.__lambda)))
        #print "alpha=",alpha
        return alpha

    def distance_from_bmu(self,bmu,current_node_vector,time_step):
        """calculate the distance between the bmu and the node"""
        
        distance = np.linalg.norm(bmu-current_node_vector, ord=2)
        radius_of_the_neighborhood = self.find_radius_of_the_neighborhood(time_step)
        theta = np.exp(-(distance/2*(radius_of_the_neighborhood)**2))
        #print "theta=",theta
        return theta

    def train(self, data):
        """train the neural network"""
        #self.array_of_node_with_weights = self.initialize_weights(self.__m,self.__n,self.__dim)

        t = 0
        while t < 100:
            print "timestep=",t
            #pick a random sample
            sample_index = np.random.randint(len(data))
            #print "for iteration ", t, " data = ", data[sample_index], "index = ", sample_index
            self.bmu,self.bmu_index = self.find_bmu(data[sample_index]) #returns index of bmu node
            #print "bmu weights=",self.bmu
            #print "bmu index=",self.bmu_index
            self.update_weights(self.bmu,data[sample_index],t)
            t += 1
        return 0
    
    def best_matching_node(self,data):
        """find the best matching unit for each passed data point"""
        dist = np.inf
        index = []
        for i in range(self.__m):
            for j in range(self.__n):
                temp_dist = 0
                temp_dist = np.linalg.norm(data-self.map_with_weight_vectors[i][j], ord=2)
                if temp_dist < dist:
                    dist = temp_dist
                    index = [i, j]
                else:
                    pass
        return self.map_with_weight_vectors[index[0]][index[1]],index
            
        

if __name__ == "__main__":
    #create the som object
    som = SOM(3, 3, 4,seed=1)
    som.print_variables()
    iris = datasets.load_iris()
    iris_data = iris.data
    #train the som model
    som.train(iris_data)
    print "original map=\n",som.original_map_weight_vectors
    print "map after training=\n",som.map_with_weight_vectors
    
    for i in range(149):
        weight, index = som.best_matching_node(iris_data[i,:])
        print "index=",index,"weight=",weight,"for data=",i
    
