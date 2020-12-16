import math


def calDim(crop_size):
    x1 = math.floor((crop_size+2-(2-1)-1)/2+1)
    x2 = math.floor((x1+2-(3-1)-1)/2+1)
    x3 = math.floor((x2+2-(3-1)-1)/2+1)
    y1 = (x3-1)*2-2+(3-1)+1+1
    y2 = (y1-1)*2-2+(2-1)+1+1
    y3 = (y2-1)*2-2+(2-1)+0+1
    print(x1, x2, x3, y1, y2, y3)
    return y3


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

available_input_sizes = []
available_residuals = []
available_residuals_divisors = []
for i in range(0, 701):
    if i == calDim(i):
        available_input_sizes.append(i)
        if isPrime(701-i) == False:
            available_residuals.append(701-i)
            available_residuals_divisors.append(get_divisor(701-i))

print(available_input_sizes)
print(available_residuals)
print(available_residuals_divisors)
calDim(356)