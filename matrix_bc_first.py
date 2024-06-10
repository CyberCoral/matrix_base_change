
import sympy, exponential_mod as ex_mod, numpy as np

##
## matrix_base_change's first iteration of the algorithm
## 


def matrix_change_base_1(n, base1, base2):
    '''
    It uses matrices to do
    the base_change algorithm.
    '''


    if not isinstance(n, list):
        #dig = int(round(math.log(n, base1),0)) + 1 
        dig = int(sympy.log(n, base2)) + 1
        n = number_to_vec(n, base1)
    else:
        if [abs(i) < abs(base1) for i in n].count(True) == 0:
            raise TypeError("Array isn't in base1.")
        n_ = check_decimal_results(n, base1)
        #dig = int(round(math.log(n_, base2),0))
        #dig = int(round(math.log(n, base1),0))

        dig = int(sympy.log(n_, base2)) + 1
        #dig = len(n)
        
    len_n = range(len(n))
    phi_b2 = ex_mod.phi(base2)

    # Array that represents the digits of a number
    # reduced by doing modulo base2.
    co = [i for i in n]
    #co = [ex_mod.exponential_mod(i, 1, base2) for i in co]

    # Coefficient matrix (the number's digits).   (order len(n) x dig)
    coef = np.array([co for i in range(dig)])

    # Array that represents the exponents of the position place of numbers
    # It gets reduced by a number made with phi(base2)
    #ex = [i for i in len_n][::-1]
    #ex = [matrix_exp_reduction(i, phi_b2 + 1) % (phi_b2 + 1) for i in ex]
    
    # Exponent matrix (order dig x dig)
    #exp = np.array([ex for j in range(dig)])

    # B1_prime matrix (a matrix with all of its elements being base1 % base2) (order len(n) x dig)
    #mat_b1_p = np.array([[matrix_exp_reduction(base1,base2) % base2 for i in len_n] for j in range(dig)])

    # B2 matrix (a matrix with all of its elements being base2) (order len(n) x dig)
    mat_b2 = np.array([[base2 for j in len_n] for i in range(dig)])

    # Reduction matrix (order len(n) x dig)
    m_redx = np.array([[sympy.Integer(base2) ** (0-i) for j in len_n] for i in range(dig)])

    # Exp-Base matrix (order len(n) x dig)
    m_ampx = np.array([[sympy.Integer(base1) ** (len(n) - i - 1) for i in len_n] for j in range(dig)])

    # Conversion factor matrix
    m_conv = m_redx * m_ampx

    # Coeficient matrix modulo mat_b2 (order len(n) x dig)
    #coef_s = (coef % mat_b2)
    coef_s = coef

    # Exponent matrix modulo mat_b2
    #exp_s = (mat_b1_p ** exp) % mat_b2

    #print("\n\n",np.array(coef_s* m_conv, np.int64),"\n\n")

    # Transformed number matrix
    try:
        mat_num = np.array(coef_s* m_conv, np.int64) #* exp_s
    except OverflowError:
        mat_num = np.array(coef_s* m_conv, sympy.Integer)
        mat_num = np.floor(mat_num, dtype=sympy.Integer)

    #print("floor(\n", coef_s,"\n * \n",m_conv,")\n  = \n", mat_num ,"\n mod \n",mat_b2,"\n\n")   

    mat_result = mat_num % mat_b2
    
    mat_result_2 = np.array([sum(mat_result[i]) for i in range(len(mat_result))][::-1],sympy.Integer)

    while [mat_result_2[i] < base2 for i in range(len(mat_result_2))].count(False) != 0:
        mat_result_2 = Div_Mod(mat_result_2, base2)
        mat_result_2 = Union(mat_result_2)
        
    return np.array(mat_result_2)

