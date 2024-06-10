# ver. Fri/23/Feb/2024
#
# Made by: CyberCoral
# ------------------------------------------------
# Github:
# https://www.github.com/CyberCoral
#

# The only use of sys.
import sys

sys.set_int_max_str_digits(1000000000)

### 
### The converter function.
###

def GeneralBaseConverter(num, original_base: int ,final_base: int):
    '''
    It converts any number of any original_base
    to another final_base, given the dictio dict
    and the lista list.
    '''

    def BaseToDecimal(num, base: int):
        '''
        It transforms a number in any base
        to decimal, given the dictionary dictio.
        '''

        lista1 = []

        if not isinstance(num, list):
            lista1 = [int(i) for i in (",".join(str(num))).split(",")[::-1]]
        else:
            lista1 = num[::-1]
                
        #####
        ##### Check if number "num" is in base "base" by checking each element.
        #####
        
        return sum([lista1[k] * base ** k for k in range(len(lista1))])


    def DecimalToBase(num, base: int):
        '''
        It converts a number in decimal form
        to any base, given the dictionary dictio
        and the list lista.
        ''' 

        lista_inf = []
        
        #####
        ##### This process is the reverse to Base2Decimal's, as it is used to know which character
        ##### or number is equivalent to from decimal to other bases.
        #####

        num = int(num)
        
        while num >= 1:
            lista_inf.append(num % base)
            num //= base
            
        lista_inf = lista_inf[::-1]

        return lista_inf
                     
    ###
    ### Main conversion section.
    ###

    num = BaseToDecimal(num, original_base)
    result = DecimalToBase(num,final_base)  
    return result
