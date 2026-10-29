def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

              import numpy as np
              try:
                return np.dot(a,b)
              except ValueError as e:
                return -1
