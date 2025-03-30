"""
This module implements functions that predict water density, specific heat, and viscosity from temperature and pressure.
"""

import tensorflow as tf


def Coef_IF97_Eq15_Region1(i):
    if i == 1:
        Ii = 0; Ji = -2; ni = 0.14632971213167
    elif i == 2:
        Ii = 0; Ji = -1; ni = -0.84548187169114
    elif i == 3:
        Ii = 0; Ji = 0; ni = -0.37563603672040*10
    elif i == 4:
        Ii = 0; Ji = 1; ni = 0.33855169168385*10
    elif i == 5:
        Ii = 0; Ji = 2; ni = -0.95791963387872
    elif i == 6:
        Ii = 0; Ji = 3; ni = 0.15772038513228
    elif i == 7:
        Ii = 0; Ji = 4; ni = -0.16616417199501*10**(-1)
    elif i == 8:
        Ii = 0; Ji = 5; ni = 0.81214629983568*10**(-3)
    elif i == 9:
        Ii = 1; Ji = -9; ni = 0.28319080123804*10**(-3)
    elif i == 10:
        Ii = 1; Ji = -7; ni = -0.60706301565874*10**(-3)
    elif i == 11:
        Ii = 1; Ji = -1; ni = -0.18990068218419*10**(-1)
    elif i == 12:
        Ii = 1; Ji = 0; ni = -0.32529748770505*10**(-1)
    elif i == 13:
        Ii = 1; Ji = 1; ni = -0.21841717175414*10**(-1)
    elif i == 14:
        Ii = 1; Ji = 3; ni = -0.52838357969930*10**(-4)
    elif i == 15:
            Ii = 2; Ji = -3; ni = -0.47184321073267*10**(-3)
    elif i == 16:
        Ii = 2; Ji = 0; ni = -0.30001780793026*10**(-3)
    elif i == 17:
        Ii = 2; Ji = 1; ni = 0.47661393906987*10**(-4)
    elif i == 18:
        Ii = 2; Ji = 3; ni = -0.44141845330846*10**(-5)
    elif i == 19:
        Ii = 2; Ji = 17; ni = -0.72694996297594*10**(-15)
    elif i == 20:
        Ii = 3; Ji = -4; ni = -0.31679644845054*10**(-4)
    elif i == 21:
        Ii = 3; Ji = 0; ni = -0.28270797985312*10**(-5)
    elif i == 22:
        Ii = 3; Ji = 6; ni = -0.85205128120103*10**(-9)
    elif i == 23:
        Ii = 4; Ji = -5; ni = -0.22425281908000*10**(-5)
    elif i == 24:
        Ii = 4; Ji = -2; ni = -0.65171222895601*10**(-6)
    elif i == 25:
        Ii = 4; Ji = 10; ni = -0.14341729937924*10**(-12)
    elif i == 26:
        Ii = 5; Ji = -8; ni = -0.40516996860117*10**(-6)
    elif i == 27:
        Ii = 8; Ji = -11; ni = -0.12734301741641*10**(-8)
    elif i == 28:
        Ii = 8; Ji = -6; ni = -0.17424871230634*10**(-9)
    elif i == 29:
        Ii = 21; Ji = -29; ni = -0.68762131295531*10**(-18)
    elif i == 30:
        Ii = 23; Ji = -31; ni = 0.14478307828521*10**(-19)
    elif i == 31:
        Ii = 29; Ji = -38; ni = 0.26335781662795*10**(-22)
    elif i == 32:
        Ii = 30; Ji = -39; ni = -0.11947622640071*10**(-22)
    elif i == 33:
        Ii = 31; Ji = -40; ni = 0.18228094581404*10**(-23)
    elif i == 34:
        Ii = 32; Ji = -41; ni = -0.93537087292458*10**(-25)
    return Ii, Ji, ni

# Region 1
def IF97_SpecificVol_Region1(Tdeg,pMPa):
    gamma_pi = 0.0
    R = 0.461526*10**(-3) # Specific gas constant [kJ/(g*K)]
    TK = Tdeg + 273
    tau = 1386/TK
    ppi = pMPa/16.53
    for i in range(1,35):
        Ii, Ji, ni = Coef_IF97_Eq15_Region1(i)
        gamma_pi = gamma_pi - ni*Ii*((7.1-ppi)**(Ii-1))*((tau-1.222)**Ji)
    svol = ppi*gamma_pi*R*TK/pMPa
    return svol

# Region 1 Specific_isobaric_heatcapacity
def IF97_SpecificCp_Region1(Tdeg,pMPa):
    gamma_tautau = 0.0
    R = 0.461526 # Specific gas constant [kJ/(kg*K)]
    TK = Tdeg + 273
    tau = 1386/TK
    ppi = pMPa/16.53
    for i in range(1,35):
        Ii, Ji, ni = Coef_IF97_Eq15_Region1(i)
        gamma_tautau = gamma_tautau + ni*((7.1-ppi)**Ii)*Ji*(Ji-1)*(tau-1.222)**(Ji-2)
    scp = -R*(tau**2)*gamma_tautau
    return scp*10**3 # [J/(kg*K)]

def Densw_pred(Tdeg,pPa):
    pMPa = pPa/(10**6)
    Densw = 1.0/IF97_SpecificVol_Region1(Tdeg,pMPa)
    return Densw

def HCw_pred(Tdeg,pPa):
    pMPa = pPa/(10**6)
    HCw_calc = IF97_SpecificCp_Region1(Tdeg,pMPa)
    return HCw_calc

def Viscow_pred(Tdeg,dens):
    TK = Tdeg + 273
    That = TK / 647.096
    cof0 = 1.67752 / That**0
    cof1 = 2.20462 / That**1
    cof2 = 0.6366564 / That**2
    cof3 = -0.241605 / That**3
    cof = cof0 + cof1 + cof2 + cof3
    myu0 = 100.0 * tf.math.sqrt(That) / cof
    H0 = [5.20094*10**(-1), 2.22531*10**(-1), -2.81378*10**(-1), 1.61913*10**(-1), -3.25372*10**(-2), 0.0, 0.0]
    H1 = [8.50895*10**(-2), 9.99115*10**(-1), -9.06851*10**(-1), 2.57399*10**(-1), 0.0, 0.0, 0.0]
    H2 = [-1.08374, 1.88797, -7.72479*10**(-1), 0.0, 0.0, 0.0, 0.0]
    H3 = [-2.89555*10**(-1), 1.26613, -4.89837*10**(-1), 0.0, 6.98452*10**(-2), 0.0, -4.35673*10**(-3)]
    H4 = [0.0, 0.0, -2.57040*10**(-1), 0.0, 0.0, 8.72102*10**(-3), 0.0]
    H5 = [0.0, 1.20573*10**(-1), 0.0, 0.0, 0.0, 0.0, -5.93264*10**(-4)]
    H = [H0, H1, H2, H3, H4, H5]
    rouhat = dens / 322.0
    myu1tp = 0.0
    for ii in range(0,6):
        Hsum = 0.0
        for jj in range(0,7):
            Hsum += H[ii][jj] * (rouhat-1)**jj
        myu1tp += ((1/That-1)**ii) * Hsum
    myu1 = tf.math.exp(rouhat*myu1tp)
    Visw_calc = myu0 * myu1 * 10**(-6)
    return Visw_calc
