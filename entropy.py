
import math
math.log2(2)

# functions:
def entropy(a,b):
    x=a/(a+b)
    y=b/(a+b)
    return (- x * math.log2(x) - y * math.log2(y))



print(entropy(549,342))
print(entropy(1,19))
print(entropy(233,81))
print(entropy(109,468))
print(entropy(372,119))
            


print(2+3)

12+45

29+46