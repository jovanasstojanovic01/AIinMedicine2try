import numpy as np
import pandas as pd

def correct_IOP(iop, cct, bazni_cct=545.0, faktor=0.05):
    # Formula: Korigovani IOP = Izmereni IOP - (CCT - Bazni_CCT) * Faktor
    korigovani_iop = iop - (cct - bazni_cct) * faktor
    
    return np.round(korigovani_iop, 2)