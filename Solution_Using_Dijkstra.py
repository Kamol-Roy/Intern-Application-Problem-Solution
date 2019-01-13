import numpy as np
from collections import defaultdict

class Directed_Graph:
    def __init__(self):
        self.nodes=set()
        self.edges=defaultdict(list)
        self.distances={}

    def add_node(self,name):
        self.nodes.append(name)
        
    def add_edge(self,from_node, to_node,distance):
        self.nodes.add(to_node)
        self.nodes.add(from_node)
        self.edges[from_node].append(to_node)

	  	
        if from_node not in self.distances:
            self.distances[from_node]={to_node:distance} 
            # if not directted there will be another line given below
            # self.distances[to_node]={from_node:distance} 

        else:
            self.distances[from_node][to_node]=distance		


def dijkstra(graph,initial_node,initial_wt):
    '''
    Given a initial node and initianl weight it will return the minimum weight 
    for all the nodes along with the last node visited. Normally initial_wt should
    be zero but in our case initial_wt = root node's weight. Beacuse we have
    shifted the triange by 1 node by assigning to_node value as edge weight. 
    '''    
    visited_path=set()
    sortest_path_wt={}
    sortest_wt=dict(zip(graph.nodes,np.repeat(np.inf,len(graph.nodes))))
    sortest_wt[initial_node]=initial_wt
            
    while initial_node:
#        print(sortest_wt,'  ', initial_node,'  ',visited_path, '\n\n')
        for e_end in graph.edges[initial_node]:
            wt=sortest_wt[initial_node]+graph.distances[initial_node][e_end]
            if wt<sortest_wt[e_end]:
                sortest_wt[e_end]=wt
                sortest_path_wt[e_end]=[initial_node,sortest_wt[e_end]]
        visited_path.add(initial_node)

        initial_node_list=[]
        for node in sorted(sortest_wt,key=sortest_wt.get,reverse=False):
            if node not in visited_path:
                initial_node_list.append(node)
        if len(initial_node_list)>=1:
            initial_node=initial_node_list[0]
        else:
            initial_node=None
#        print(initial_node_list)
         
    return visited_path,sortest_wt, sortest_path_wt
    

def Graph_input(data):
    '''
    converting the triangle as an directed graph, where the coordinates
    of triangle vertexs are considered as node and the vertex values
    are assigned as values of the connecting nodes. In the directed
    connection(from_node - to_node) to_node value is assigned as the
    connecting edge value
    
    '''    
    data_mat=[[int(x) for x in row.strip().split("  ")] for row in data] #integer conversion               
 
    num_dict={}           
    for row in range(len(data_mat)):
        for column in range(len(data_mat[row])):
            # Mapping the vertex values to their co-ordinates
            num_dict[(row,column)]=data_mat[row][column]
    
    for row_index in range(len(data)):
        if row_index<len(data)-1:
            nodes=data_mat[row_index]                    
            for i in range(len(nodes)):
                from_node=(row_index,i) # For each node there are two 
                to_node=(row_index+1,i) #possible conncetions 
                to_node_1=(row_index+1,i+1)
                
                graph.add_edge(from_node, to_node, num_dict[to_node]) # to_node value is assigned as edge value
                graph.add_edge(from_node, to_node_1, num_dict[to_node_1])
    return graph,num_dict
    
def Minimal_path(graph,shortest_path_wt):
    
    '''
    Dijkstra function gives all the shortest weight for each node. But we
    are interested only minimum weight for the last row.
    '''
    total_sum=np.inf
    least_node=''
    for node in graph.nodes:
        x_cord,y_cord=node

        if x_cord==len(data)-1:
            if shortest_path_wt[node][1]<total_sum:
                total_sum=shortest_path_wt[node][1]
                least_node=node
                
        lt=[] #To get all the node it has visited to reach the targeted node 
             #because shortest_path_wt tracks only the last node it has visited. 
        
    i=0
    subkey=least_node
    while i<len(data):
        if subkey !=(0,0):                      
            key=shortest_path_wt[subkey][0]
            lt.append(key)
            subkey=key
        i+=1

        
              
    value_path=[num_dict[i] for i in ([least_node]+lt)[::-1]]
    return least_node, value_path, total_sum
    
if __name__ == "__main__":

#    import sys
#    print (sys.argv)
#    
    graph=Directed_Graph()
#    file_input = sys.argv[1]
    file_input= "C:\intern_min_path\large_triangle.txt"    
    filename= open(file_input,'r')
    data=filename.readlines()
    filename.close()
    
    graph,num_dict=Graph_input(data)
    
    initial_point=(0,0)
    initial_wt=num_dict[initial_point]    
    visited_path,sortest_wt, sortest_path_wt =dijkstra(graph,initial_point,initial_wt)


    least_node, value_path, total_sum= Minimal_path(graph,sortest_path_wt)
    
    print("Path : ", value_path,'\nTotal Sum : ',total_sum)
    
    
    
    
    