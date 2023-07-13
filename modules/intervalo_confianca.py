import numpy as np
from math import sqrt

def intervalo_de_confianca(array, Zc=1.96):
    desvio_padrao = np.std(array)
    margem_de_erro = Zc*(desvio_padrao/sqrt(len(array)))

    media = sum(array)/len(array)

    intervalo = [media-margem_de_erro, media+margem_de_erro]
    
    return intervalo
