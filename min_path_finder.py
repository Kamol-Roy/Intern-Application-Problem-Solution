filename= open("C:\intern_min_path\small_triangle.txt",'r')
data=filename.readlines()
filename.close()
import numpy as np



triangle={}
for row_index in range(len(data)):
    nodes=data[row_index].strip().split("  ")
    values=data[row_index+1].strip().split("  ")
    
    for i in range(len(nodes)):
        if row_index not in triangle:
            triangle[row_index]={int(nodes[i]):[int(values[i]),int(values[i+1])]}

        else:
    
            triangle[row_index][int(nodes[i])]=[int(values[i]),int(values[i+1])]

mod_triangle={}
for row_index in range(len(data)):
    nodes=data[row_index].strip().split("  ")
    values=data[row_index+1].strip().split("  ")
    
    for i in range(len(nodes)):
        if row_index not in triangle:
            triangle[row_index]={int(nodes[i]):[int(values[i]),int(values[i+1])]}

        else:
    
            triangle[row_index][int(nodes[i])]=[int(values[i]),int(values[i+1])]




def Minimum_Path(triangle):
    
    path=[]
    rows=[]

    for i in triangle:
        if i not in rows:
            for j in triangle[i]:
                path.append([{i:j}])
            rows.append(i)
            

        
        
    
    
    
    
    




if __name__ == "__main__":
    
    Minimum_Path()
