import numpy as np
file_input= "C:\intern_min_path\large_triangle.txt"    
filename= open(file_input,'r')
data=filename.readlines()
filename.close()

data_mat=[[int(x) for x in row.strip().split("  ")] for row in data] #integer conversion               
 
#num_dict={}           
#for row in range(len(data_mat)):
#    for column in range(len(data_mat[row])):
#        # Mapping the vertex values to their co-ordinates
#        num_dict[(row,column)]=[data_mat[row][column],data_mat[row][column],0]

num_dict={(row,column):[data_mat[row][column],data_mat[row][column],0] for row 
          in range(len(data_mat)) for column in range(len(data_mat[row]))}

for row in range(len(data_mat)-1):

    for column1 in range(len(data_mat[row])):
        
        if column1==0:
            num_dict[(row+1,column1)][1]+=num_dict[(row,column1)][1]
            num_dict[(row+1,column1)][2]=(row,column1)

            if column1+1==len(data_mat[row+1])-1:
                num_dict[(row+1,column1+1)][1]+=num_dict[(row,column1)][1]
                num_dict[(row+1,column1+1)][2]=(row,column1)

        else:

            sum0=num_dict[(row+1,column1)][1]+num_dict[(row,column1-1)][1]
            sum1=num_dict[(row+1,column1)][1]+num_dict[(row,column1)][1]
            if sum0<sum1:
                num_dict[(row+1,column1)][1]=sum0
                num_dict[(row+1,column1)][2]=(row,column1-1)

            else:
                num_dict[(row+1,column1)][1]=sum1
                num_dict[(row+1,column1)][2]=(row,column1)
            if column1+1==len(data_mat[row+1])-1:
                num_dict[(row+1,column1+1)][1]+=num_dict[(row,column1)][1]
                num_dict[(row+1,column1+1)][2]=(row,column1)
                


least_node=''
total_sum= np.inf
for node in num_dict.keys():
    x_cord,y_cord=node

    if x_cord==len(data)-1:
        if num_dict[node][1]<total_sum:
            total_sum=num_dict[node][1]
            least_node=node
            
lt=[] 
i=0
subkey=least_node
while i<len(data):                     
    key=num_dict[subkey][2]
    lt.append(key)
    subkey=key
    i+=1

      
value_path=[num_dict[i][0] for i in ([least_node]+lt[:-1])[::-1]]                                



print('Path :', value_path,'\nTotal Sum :', total_sum )


                





