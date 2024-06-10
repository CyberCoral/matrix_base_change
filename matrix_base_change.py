###
### Name of project: matrix_base_change.py
###
### Author: CyberCoral
###
### Purpose of project: A conversion algorithm for vectors and matrices,
###                                 which is done with numpy and sympy for accuracy and speed,
###                                 plus threading.
###
### Link of project in Github:
### https://github.com/CyberCoral/matrix_base_change
###


import numpy as np, sympy.ntheory as synth, math, sympy
import threading, os
import sympy.ntheory as synth

# A constant for threaded change_base
val = 1 / 2

with open("requirements.txt","r") as f:
    a =[f.readlines()][0]
    a = [(lambda b,c: c[b][::-1][1:][::-1] if b != len(c) - 1 else c[b])(i, a) for i in range(len(a))]

    import importlib
    for i in range(len(a)):
        loader = importlib.util.find_spec(a[i])
        if loader == None:
            os.system(f"python -m pip install {a[i]}")

def scan_list(x):
    if not isinstance(x, list):
        return x
    
    scan = []
    def scan_(x):
        if not isinstance(x, list):
            return x
        for i in range(len(x)):
            if isinstance(x[i], list):
                scan_(x[i])
            else:
                scan.append(x[i])
    scan_(x)
    return scan

def expanded_factors(x: int) -> list:
    '''
    It represents the factors
    of a number as a complete list,
    instead of a dictionary.
    '''
    
    return scan_list([[k]*v for k,v in synth.factorint(x).items()])

def factors(a: int) -> dict:
    '''
    Returns prime factors of 
    the integer a.
    '''
    return synth.factorint(a)

def a_contains_b(a: list, b: list) -> bool:
    '''
    Checks if "a" contains or
    is equal to "b".
    '''
    c, d = [i for i in a], [i for i in b]
    while True:
        if len(d) == 0:
            return True
        elif d[0] in c:
            del c[c.index(d[0])]
            del d[0]
        else:
            return False

def n_multiple_of(a: int, b: int) -> int:
    '''
    It checks if the integer forms
    of "a" and "b" meet the relation
    a^n mod b = 0, determining 
    the n value.
    '''
    if not a_contains_b(expanded_factors(a),expanded_factors(b)) or b in [0,1]:
        return 0
    
    a, b = factors(a), factors(b)
    
    key_b = list(b.keys())
    
    occurences = [a[i] // b[i] for i in key_b]
    
    n_mult = min(occurences)
    
    return n_mult

def number_to_vec(n, base):
    return [(n // (base ** i)) % base for i in range(int(round(math.log(n, base),0)) + 1)][::-1]

def matrix_exp_reduction(n, redx):
    '''
    Reduces the elements of the array by redx,
    uses exponential_mod functions and sympy.
    '''
    n_mult =  n_multiple_of(n, redx)
    return n // sympy.gcd(n, redx ** n_mult)

def check_decimal_results(arr, base):
    return sum([arr[::-1][i] * base ** i for i in range(len(arr))])

def Div_Mod(arr, m):
    return np.array([[arr[i] // m, arr[i] % m] for i in range(len(arr))])

def Union(arr):
    return np.array([arr[i][1] + arr[i + 1][0] for i in range(0, len(arr) - 1)] + [arr[::-1][0][1]],sympy.Integer)

def mod(a, m):
    '''
    Does
    a mod m,
    but with some conditions.
    '''
    try:
        if list(abs(a) < abs(m)).count(False) == 0:
            return a
    except TypeError:
        if abs(a) < abs(m):
            return a
    return a % m

def vector_change_base(n, base2):
    '''
    It uses vectors to do
    the base_change algorithm.

    n is in base 10.
    '''

    dig = sympy.log(n, base2) + 1

    # Coefficient vector.
    coef = np.array([n for i in range(dig)])

    # Base2 vector.
    vec_b2 = np.array([base2 for i in range(dig)])

    # Conversion vector.
    vec_conv = [sympy.Integer(base2) ** (-1*i) for i in range(dig)]
    vec_conv = np.array(vec_conv)

    try:
        vec_num = np.array(coef* vec_conv, np.int64) #* exp_s
    except OverflowError:
        vec_num = np.array(coef* vec_conv, sympy.Integer)
        vec_num = np.floor(vec_num, dtype=sympy.Integer)

    #print("floor(\n", coef,"\n * \n",vec_conv,")\n  = \n", vec_num ,"\n mod \n",vec_b2,"\n\n")   

    # Result vector.
    vec_result = mod(vec_num[::-1], vec_b2)

    return vec_result

# WIP
def threaded_vector_change_base_1(n, base2,*,val: float = val):
    '''
    It mixes Threads (with threading library)
    and vector_base_change to
    create a faster version of
    base change algorithm.

    This one defines the arrays first
    and then they are sliced.

    The number is, by default, in base 10.
    '''
    dig = sympy.Integer(sympy.log(n, base2)) + 1

    arr = []
    # Constants.
    special_num = sympy.Integer(dig ** val)
    
    mod_num = dig % special_num
    remainder = dig // special_num

    # Coefficient vector.
    coef = np.array([n for i in range(dig)])

    # Base2 vector.
    vec_b2 = np.array([base2 for i in range(dig)])

    # Conversion vector.
    vec_conv = [sympy.Integer(base2) ** (-1*i) for i in range(dig)]
    vec_conv = np.array(vec_conv)

    
    def f(t, total, i):
        arr.append(mod(np.floor(np.array(coef[total*t:(t+1) * total - i] * vec_conv[total*t:(t+1) * total - i], sympy.Integer), dtype=sympy.Integer), vec_b2[total*t:(t+1) * total - i]))

    for j in range(special_num):
        a = threading.Thread(target = f(j,remainder,0))

    if mod_num != 0:
        f(j+1,remainder, remainder - mod_num)

    vec_res = []
    k = 0

    while k < len(arr):
        vec_res += list(arr[k])
        k += 1

    return np.array(vec_res[::-1], sympy.Integer)

def threaded_vector_change_base_2(n, base2,*,val: float = val):
    '''
    It mixes Threads (with threading library)
    and vector_base_change to
    create a faster version of
    base change algorithm.

    This one defines the variable arrays
    when they are
    invoked by the thread.

    The number is, by default, in base 10.
    '''
    dig = sympy.Integer(sympy.log(n, base2)) + 1

    arr = []
    # Constants.
    special_num = sympy.Integer(dig ** val)
    
    mod_num = dig % special_num
    remainder = dig // special_num
    
    def f(t, total, i):
        coef = np.array([n for i in range(total*t,(t+1) * total - i)])
        vec_b2 = np.array([base2 for i in range(total*t,(t+1) * total - i)])
        vec_conv = np.array([sympy.Integer(base2) ** (-1*i) for i in range(total*t,(t+1) * total - i)])

        arr.append(mod(np.floor(np.array(coef * vec_conv, sympy.Integer), dtype=sympy.Integer), vec_b2))

    for j in range(special_num):
        a = threading.Thread(target = f(j,remainder,0))

    if mod_num != 0:
        f(j+1,remainder, remainder - mod_num)

    vec_res = []
    k = 0

    while k < len(arr):
        vec_res += list(arr[k])
        k += 1

    return np.array(vec_res[::-1], sympy.Integer)
        
def rank_2_tensor_change_base(n: list, base2, *, array_: bool = True, thread_: bool = False):
    '''
    It uses scalars to
    represent the different numbers,
    and it applies vector_base_change
    to do the base_change algorithm
    for each of them.

    n's elements are, by default, in base 10.
    '''

    if not isinstance(n, list) and not isinstance(n, np.array):
        raise TypeError("n must be either a list or an np.array")
    elif not isinstance(n, list):
        n = list(n)

    k = 1

    vector = (lambda t: threaded_vector_change_base_1 if t == True else vector_change_base)(thread_)

    T_result = [vector(n[i], base2) for i in range(len(n))]
    m_result = (lambda a: list(T_result[0]) if a == False else [T_result[0]])(array_)

    while k < len(T_result):
        m_result += (lambda a: list(T_result[k]) if a == False else [T_result[k]])(array_)
        k += 1

    return m_result

def vector_to_vector_change_base(n, base1, base2):
    '''
    It transforms a vector n
    to scalar and then applies
    vector_change_base to it.
    '''
    n = check_decimal_results(n, base1)
    return vector_change_base(n, base2)

def vector_to_threaded_vector_change_base(n, base1, base2):
     '''
     It transforms a vector n
     to scalar and then applies
     threaded_vector_change_base to it.
     '''
     n = check_decimal_results(n, base1)
     return threaded_vector_change_base_1(n, base2)

def matrix_change_base_1(n, base1, base2):
    '''
    It uses matrices to do
    the base_change algorithm.
    '''


    if not isinstance(n, list):
        dig = sympy.Integer(sympy.log(n, base2)) + 1
        n = number_to_vec(n, base1)
    else:
        if [abs(i) < abs(base1) for i in n].count(True) == 0:
            raise TypeError("Array isn't in base1.")
        n_ = check_decimal_results(n, base1)
        dig = sympy.Integer(sympy.log(n_, base2)) + 1
        
    len_n = range(len(n))

    # Array that represents the digits of a number
    # reduced by doing modulo base2.
    co = [i for i in n]

    # Coefficient matrix (the number's digits).   (order len(n) x dig)
    coef = np.array([co for i in range(dig)])
    
    # B2 matrix (a matrix with all of its elements being base2) (order len(n) x dig)
    mat_b2 = np.array([[base2 for j in len_n] for i in range(dig)])

    # Reduction matrix (order len(n) x dig)
    m_redx = np.array([[sympy.Integer(base2) ** (0-i) for j in len_n] for i in range(dig)])

    # Exp-Base matrix (order len(n) x dig)
    m_ampx = np.array([[sympy.Integer(base1) ** (len(n) - i - 1) for i in len_n] for j in range(dig)])

    # Conversion factor matrix
    m_conv = m_redx * m_ampx
    # Transformed number matrix
    try:
        mat_num = np.array(coef* m_conv, np.int64) #* exp_s
    except OverflowError:
        mat_num = np.array(coef* m_conv, sympy.Integer)
        mat_num = np.floor(mat_num, dtype=sympy.Integer)

    mat_result = mat_num % mat_b2
    
    mat_result_2 = np.array([sum(mat_result[i]) for i in range(len(mat_result))][::-1],sympy.Integer)

    while [mat_result_2[i] < base2 for i in range(len(mat_result_2))].count(False) != 0:
        mat_result_2 = Div_Mod(mat_result_2, base2)
        mat_result_2 = Union(mat_result_2)
        
    return np.array(mat_result_2)

def matrix_change_base_2(n, base1, base2):
    '''
    It uses matrices to do
    the base_change algorithm.
    This one is experimental, it does not
    work with big numbers ( > 10 ** 1000).
    '''


    if not isinstance(n, list):
        dig = sympy.Integer(sympy.log(n, base2)) + 1
        n = number_to_vec(n, base1)
    else:
        if [abs(i) < abs(base1) for i in n].count(True) == 0:
            raise TypeError("Array isn't in base1.")
        n_ = check_decimal_results(n, base1)
        dig = sympy.Integer(sympy.log(n_, base2)) + 1


    # Variables    
    len_n = range(len(n))
    b1_const = sympy.log(base1, base2)   
    dig_const = len(n)

    # Array that represents the digits of a number
    # reduced by doing modulo base2.
    co = [i for i in n]

    # Coefficient matrix (the number's digits).   (order len(n) x dig)
    coef = np.array([co for i in range(dig)])

    # B2 matrix (a matrix with all of its elements being base2) (order len(n) x dig)
    mat_b2 = np.array([[base2 for j in len_n] for i in range(dig)])
    
    # Conversion matrix (order len(n) x dig)
    # It reduces the coefficients based on base1's powers and base2's inverse powers.
    m_conv = [[ sympy.Integer(base2) ** ((-1*i + ((dig_const) -j  - 1) * b1_const)) for j in len_n] for i in range(dig)]
    m_conv = np.array(m_conv)

    try:
        mat_num = np.array(coef* m_conv, np.int64) #* exp_s
    except OverflowError:
        mat_num = np.array(coef* m_conv, sympy.Integer)
        mat_num = np.floor(mat_num, dtype=sympy.Integer)

    mat_result = mat_num % mat_b2
    
    mat_result_2 = np.array([sum(mat_result[i]) for i in range(len(mat_result))][::-1],sympy.Integer)

    while [mat_result_2[i] < base2 for i in range(len(mat_result_2))].count(False) != 0:
        mat_result_2 = Div_Mod(mat_result_2, base2)
        mat_result_2 = Union(mat_result_2)

    #print("floor(\n", coef,"\n * \n",m_conv,")\n  = \n", np.array(coef* m_conv, sympy.Integer) ,"\n mod \n",mat_b2,"\n = \n",mat_result,"\n\n")


    return mat_result_2

def threaded_matrix_change_base(n, base1, base2, val=val):
    '''
    It It mixes Threads (with threading library)
    with  matrices to make
    the base_change algorithm go faster.
    '''

    if not isinstance(n, list):
        dig = sympy.Integer(sympy.log(n, base2)) + 1
        n = number_to_vec(n, base1)
    else:
        if [abs(i) < abs(base1) for i in n].count(True) == 0:
            raise TypeError("Array isn't in base1.")
        n_ = check_decimal_results(n, base1)
        dig = sympy.Integer(sympy.log(n_, base2)) + 1
        
    len_n = range(len(n))

    arr = []
    # Constants.
    special_num = sympy.Integer(dig ** val)
    
    mod_num = dig % special_num
    remainder = dig // special_num

    # Array that represents the digits of a number
    # reduced by doing modulo base2.
    co = [i for i in n]

    # Coefficient matrix (the number's digits).   (order len(n) x dig)
    coef = np.array([co for i in range(dig)])
    
    # B2 matrix (a matrix with all of its elements being base2) (order len(n) x dig)
    mat_b2 = np.array([[base2 for j in len_n] for i in range(dig)])

    # Reduction matrix (order len(n) x dig)
    m_redx = np.array([[sympy.Integer(base2) ** (0-i) for j in len_n] for i in range(dig)])

    # Exp-Base matrix (order len(n) x dig)
    m_ampx = np.array([[sympy.Integer(base1) ** (len(n) - i - 1) for i in len_n] for j in range(dig)])

    def f(t, total, i):
        m_conv = m_redx[total*t:(t+1) * total - i] * m_ampx[total*t:(t+1) * total - i]

        num = np.floor(np.array(coef[total*t:(t+1) * total - i] * m_conv, sympy.Integer), dtype=sympy.Integer) % mat_b2[total*t:(t+1) * total - i]
        arr.append([sum(num[i]) for i in range(len(num))])

    for j in range(special_num):
        a = threading.Thread(target = f(j,remainder,0))

    if mod_num != 0:
        f(j+1,remainder, remainder - mod_num)

    mat_result = []

    k = 0

    while k < len(arr):
        mat_result += list(arr[k])
        k += 1

    mat_result = mat_result[::-1]

    while [mat_result[i] < base2 for i in range(len(mat_result))].count(False) != 0:
        mat_result = Div_Mod(mat_result, base2)
        mat_result = Union(mat_result)
        
    return np.array(mat_result, sympy.Integer)

def rank_3_tensor_change_base(n: list, base1, base2,*, array_: bool = True, thread_: bool = False):
    '''
    It uses vectors to
    represent the different numbers,
    and it applies matrix_base_change_2
    to do the base_change algorithm
    for each of them.
    '''

    if not isinstance(n, list) and not isinstance(n, np.array):
        raise TypeError("n must be either a list or an np.array")
    elif not isinstance(n, list):
        n = list(n)

    k = 1

    matrix = (lambda t: threaded_matrix_change_base if t == True else matrix_change_base_1)(thread_)

    T_result = [matrix(n[i], base1, base2) for i in range(len(n))]
    m_result = (lambda a: list(T_result[0]) if a == False else [T_result[0]])(array_)

    while k < len(T_result):
        m_result += (lambda a: list(T_result[k]) if a == False else [T_result[k]])(array_)
        k += 1

    return m_result


