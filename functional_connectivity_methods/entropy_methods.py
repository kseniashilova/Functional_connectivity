import numpy as np
from pyinform import transferentropy
from pyinform import blockentropy

# Example time series data
# xs = [0,1,1,1,1,0,0,0,0]
# ys = [0,0,1,1,1,1,0,0,0]

def tr_ent(data1, data2, k):
    return transferentropy.transfer_entropy(data1, data2, k=k)


def block_ent(data1, data2, k):
    return blockentropy.block_entropy(data1, data2, k)