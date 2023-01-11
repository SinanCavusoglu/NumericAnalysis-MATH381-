# For the run code, please delete """ above and below code(for which iteration you want to run)
import numpy as np
from numpy.linalg import inv
from tabulate import tabulate
start_vec1 = np.array([1,1,1,1]).reshape((4,1)) # x_0 = [1,1,1,1]
aug_mat1 = np.array([[2,-1,2,-1,-1],   # [x,y,z,w,b] Augmented Matrix
                    [2,1,-2,-2,-2],
                    [-1,2,-4,1,1],
                    [3,0,0,-3,-3]])
headers=["vec1","vec2","vec3","vec4","vec5","vec6","vec7","vec8","vec9","vec10"]
relex_headers = ["Rele1","Rele2","Rele3","Rele4","Rele5","Rele6","Rele7","Rele8","Rele9","Rele10"]
def jacobi_iterative_method(n,aug_mat,start_vec):
    """
    n: The number of equations n
    aug_mat: The augmented matrix
    start_vec: The starting vector
    """
    import numpy as np
    from numpy.linalg import inv

    coeff_mat = aug_mat[:,0:n]  # Coefficient Matrix
    b = aug_mat[:,n:]           # constant

    diag = np.diag(np.diag(coeff_mat)) # diagonal matrix
    diag_inv = inv(diag)               # inverse of diagonal matrix 
    
    x_k = ((np.identity(n) - diag_inv @ coeff_mat) @ start_vec) + (diag_inv @ b)  # @ is matrix multiplication
    relex = abs(x_k - start_vec).max() / abs(x_k).max()                            # Relative Error 
    
    return x_k, relex


def GaussSeidel_iterative_method(n,aug_mat,start_vec):
    """
    n: The number of equations n
    aug_mat: The augmented matrix
    start_vec: The starting vector
    """
    import numpy as np
    from numpy.linalg import inv

    coeff_mat = aug_mat[:,0:n]  # Coefficient Matrix
    b = aug_mat[:,n:]           # constant

    lower_tri = np.tril(coeff_mat, k = -1)  # Lower Triangular Matrix
    diag = np.diag(np.diag(coeff_mat))      # Diagonal Matrix
    upper_tri = np.triu(coeff_mat, k = 1)   # Upper Triangular Matrix

   
    x_k = inv(diag + lower_tri) @ ((- upper_tri @ start_vec) + b) # @ is matrix multiplication
    relex = abs(x_k - start_vec).max() / abs(x_k).max()           # Relative Error 
        
    return x_k, relex

"""
# Jacobi Method:
vector_list= []
rele_list = []
for k in range(1,1100):
    (start_vec1, relex) = jacobi_iterative_method(4,aug_mat1,start_vec1)
    vector_list.append(start_vec1)
    rele_list.append(relex)
    print("iter_{}".format(k))
    print(start_vec1)

result = np.hstack(vector_list)
print(tabulate(result, headers, tablefmt="github"))
print(rele_list)
"""

# GaussSeidel_iterative
"""
liste = []
rele_list = []
for k in range(1,11):
    (start_vec1, relex) = GaussSeidel_iterative_method(4,aug_mat1,start_vec1)
    liste.append(start_vec1)
    rele_list.append(relex)
result = np.hstack(liste)
print (tabulate(result, headers, tablefmt="github"))
print(np.around(np.array(rele_list), 4)) 
    """
# With Tolerance 
aug_mat2 = np.array([[1,0,1,2],
                     [1,-1,0,0],
                     [1,2,-3,0]])

start_vec2 = np.array([0,0,0]).reshape((3,1))

tolerance = 10 ** (-3)
relex_tolerance = 1000 # relative error for pass tolerance

"""
k = 1  # index for jacobi
while True:
    (start_vec2, relex) = jacobi_iterative_method(3,aug_mat2,start_vec2)
    print("Iteration {}".format(k))
    print("x_{}".format(k))
    print(start_vec2)
    print("Relative Error: {}".format(relex))
    
    relex_tolerance = relex
    k += 1
    if abs(relex_tolerance) < tolerance:
        break

"""

"""
j = 1
while True:
    (start_vec2, relex) = GaussSeidel_iterative_method(3,aug_mat2,start_vec2)
    print("Iteration {}".format(j))
    print("x_{}".format(j))
    print(start_vec2)
    print("Relative Error: {}".format(relex))
    
    relex_tolerance = relex
    j += 1
    if abs(relex_tolerance) < tolerance:
        break
    
    if j == 10000:
        break

    """



