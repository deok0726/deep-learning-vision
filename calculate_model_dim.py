import math


# def calDim(crop_size):
#     x1 = math.floor((crop_size+2-(4-1)-1)/2+1)
#     x2 = math.floor((x1+2-(4-1)-1)/2+1)
#     x3 = math.floor((x2+2-(4-1)-1)/2+1)
#     # y1 = (x3-1)*2-2+(3-1)+1+1
#     # y2 = (y1-1)*2-2+(2-1)+1+1
#     # y3 = (y2-1)*2-2+(2-1)+0+1
#     y1 = math.floor((x3+2-(3-1)-1)/1+1)*2
#     y2 = math.floor((y1+2-(3-1)-1)/1+1)*2
#     y3 = math.floor((y2+2-(3-1)-1)/1+1)*2
#     print(crop_size, x1, x2, x3, y1, y2, y3)
#     return y3

def calDim(crop_size):
    x1 = math.floor((crop_size+2-(3-1)-1)/2+1)
    x2 = math.floor((x1+2-(3-1)-1)/2+1)
    x3 = math.floor((x2+2-(3-1)-1)/2+1)
    x4 = math.floor((x3+2-(3-1)-1)/2+1)
    y1 = (x4-1)*2-2+(3-1)+1+1
    y2 = (y1-1)*2-2+(3-1)+1+1
    y3 = (y2-1)*2-2+(3-1)+1+1
    y4 = (y3-1)*2-2+(3-1)+1+1
    print(crop_size, x1, x2, x3, x4, y1, y2, y3, y4)
    return y4

def isPrime(num):
    if num == 1:
        return False
    n = int(math.sqrt(num))
    for k in range(2, n+1):
        if num % k == 0:
            return False
    return True

def get_divisor(n):
    n = int(n)
    divisors = []
    divisors_back = [] 

    for i in range(1, int(n**(1/2)) + 1): 
        if (n % i == 0):            
            divisors.append(i)
            if (i != (n // i)): 
                divisors_back.append(n//i)

    return divisors + divisors_back[::-1]

width = 701
available_input_sizes = []
available_residuals = []
available_residuals_divisors = []
available_size_residuals_and_divisors = []
for i in range(0, width):
    if i == calDim(i):
        available_input_sizes.append(i)
        if isPrime(width-i) == False:
            available_residuals.append(width-i)
            available_residuals_divisors.append(get_divisor(width-i))
            available_size_residuals_and_divisors.append([i, width-i, get_divisor(width-i)])

# print('available_input_sizes: ', available_input_sizes)
# print('available_residuals: ', available_residuals)
# print('available_residuals_divisors: ', available_residuals_divisors)
print('available_size_residuals_and_divisors: ', available_size_residuals_and_divisors)
# calDim(356)