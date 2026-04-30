import math
data = [10, 12, 15, 18, 20]
dataLength = len(data)
mean = sum(data )/ dataLength
sortedData = sorted(data)
mid = math.ceil(dataLength / 2)
median = (sortedData[mid] + sortedData[mid - 1]) / 2  if mid & 1 == 0  else sortedData[mid - 1]
dataWithOutDuplicates = list(set(data))
mode = {0:0}

for i in range(0, len(dataWithOutDuplicates)) :
    count = 0
    for j in range(0, len(data)) :
        if(dataWithOutDuplicates[i] == data[j]) :
            count += 1
    if list(mode.values())[0] < count :
        mode = {dataWithOutDuplicates[i] : count} 
    
sd = 0
for i in range(0, dataLength) :
    diff = (data[i] - mean)
    sd = sd + (diff * diff)
sd = sd / dataLength
print('sd', sd)

variance = 0
for element in data :
    variance += (element - mean) ** 2

variance = variance / dataLength

print('standard deviation:', math.sqrt(variance))

print(mean)
print(median)
print(mode)
print(sd)
print(variance)