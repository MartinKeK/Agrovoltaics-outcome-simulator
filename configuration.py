# N719 DYE ABSORTANCE FUNCTION


def trans_spectrum719(x):
    maximum_absortance = 0.6
    maximum_absortance_ref = 0.2943
    a1 =      maximum_absortance 
    b1 =       396.6
    c1 =       58.72
    a2 =      maximum_absortance
    b2 =       513.5
    c2 =       68.24  
    return (0.95 - (a1 * np.exp(-((x-b1)/c1)**2) + a2 * np.exp(-((x-b2)/c2)**2)))

