filename= open("C:\intern_min_path\small_triangle.txt",'r')
data=filename.readlines()
filename.close()
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
        else:
            self.distances[from_node][to_node]=distance		


def dijkstra(graph,initial_node):
    visited_path=set()
    sortest_path_wt={}
    sortest_wt=dict(zip(graph.nodes,np.repeat(np.inf,len(graph.nodes))))
    sortest_wt[initial_node]=79
            
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
    
    
    
graph=Directed_Graph()

triangle={}
for row_index in range(len(data)):
    if row_index<len(data)-1:
        nodes=data[row_index].strip().split("  ")
        values=data[row_index+1].strip().split("  ")
        
        for i in range(len(nodes)):
            if row_index not in triangle:
                triangle[row_index]={int(nodes[i]):[int(values[i]),int(values[i+1])]}
    
            else:
        
                triangle[row_index][int(nodes[i])]=[int(values[i]),int(values[i+1])]    
    



for row in triangle:
    if row==0:
        for i in triangle[row]:    
            from_node=str(row)
            to_node=str(row)+"_"+str(i)
            graph.add_edge(from_node,to_node,i)
        
            for j in triangle[row][i]:
                from_node=str(row)+'_'+str(i)_'_'+str(i)
                to_node=str(row)+'_'+str(i)+'_'+str(j)
                print(from_node, to_node,j)
                graph.add_edge(from_node,to_node,j)
    else:
        for i in triangle[row]:
            for j in triangle[row][i]:
                from_node=str(row-1)+'_'+str(i)
                to_node=str(row)+'_'+str(i)+'_'+str(j)
                print(from_node, to_node,j)
                graph.add_edge(from_node,to_node,j)            

            
            
p,x,y=dijkstra(graph,'0')

path_dict={}

for node in graph.nodes:
#    print(node)
    lt=[]
    i=0
    subkey=node
    while i<len(graph.nodes):
        try:
            
            key=y[subkey][0]
            lt.append(key)
            subkey=key
            i+=1
        except:
            i+=1
          
    if node not in path_dict:
        path_dict[node]=lt

        
print(path_dict)             