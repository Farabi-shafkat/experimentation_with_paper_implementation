
import random
myMap = {}
loop = 1

while loop<20:
    tmp = str(randint(0,20))
    
    if tmp in myMap:
        continue
    
    myMap[tmp] = '1'
    f.write(tmp)
    f.write(" ")
    loop = loop + 1