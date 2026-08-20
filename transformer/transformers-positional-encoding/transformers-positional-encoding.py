import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    def sinpos(pos,i):
        return np.sin(pos/(10000**(i/d_model)))

    def cospos(pos,i):
        return np.cos(pos/(10000**((i-1)/d_model)))
    
    output=[]

    for pos in range(seq_length):
        temp=[]
        for i in range(d_model):
            if i%2==0:
                temp.append(sinpos(pos,i))
            else:
                temp.append(cospos(pos,i))
        output.append(temp)

    return np.array(output)
                