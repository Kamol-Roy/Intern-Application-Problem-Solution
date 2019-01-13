import csv
import random

lines = open("large_triangle.txt").read().splitlines()

lineno = 0
cur_index = 0
path=[]

for line in lines:
      splits = line.split("  ")
      if lineno == 0:
            print (splits[cur_index])
            path.append(splits[cur_index])
      else: 
          num1 = int(splits[cur_index])
          num2 = int(splits[cur_index + 1])

          if num1 < num2:
              print (num1)
              path.append(num1)
          else:
              print (num2)
              path.append(num2)
              cur_index = cur_index + 1	
      lineno = lineno + 1		