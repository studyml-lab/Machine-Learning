import math

# entropy = - p log p - q log q

p = 549 / (549 + 342)

q = 342 / (549 + 342)

entropy = -p*math.log2(p) - q*math.log2(q)
