# %%
import sympy as sp
import numpy as np
from sympy.physics.quantum import TensorProduct
i = sp.I

# %%
class Q:
    X = sp.Matrix([
        [0, 1],
        [1, 0]
    ])
    Y = sp.Matrix([
        [0, -i],
        [i, 0]
    ])
    Z = sp.Matrix([
        [1, 0],
        [0, -1]
    ])
    H = sp.Matrix([
        [1, 1],
        [1, -1]
    ])/sp.sqrt(2)
    T = sp.Matrix([
        [1, 0],
        [0, sp.exp(i*sp.pi/4)]
    ])
    CX = sp.Matrix([
        [sp.eye(2), sp.zeros(2)],
        [sp.zeros(2), X]
    ])
    CU = lambda U: sp.Matrix([
        [sp.eye(2), sp.zeros(2)],
        [sp.zeros(2), U]
    ])
    CCNOT = sp.Matrix([
        [sp.eye(4), sp.zeros(4)],
        [sp.zeros(4), CX]
    ])
    SWAP = sp.Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    P = lambda phi: sp.Matrix([
        [1, 0],
        [0, sp.exp(i*phi)]
    ])
    XC = SWAP@CX@SWAP
    UC = lambda U: Q.SWAP@Q.CU(U)@Q.SWAP
    I = sp.eye(2)
    Rx = lambda theta: sp.Matrix([
        [sp.cos(theta/2), -i*sp.sin(theta/2)],
        [-i*sp.sin(theta/2), sp.cos(theta/2)]
    ])
    S = sp.Matrix([
        [1, 0],
        [0, i]
    ])
    Sx = sp.Matrix([
        [1+i, 1-i],
        [1-i, 1+i]
    ])/2

def swap23(U):
    sw = TensorProduct(Q.I, Q.I, Q.SWAP)
    return sw * U * sw

def Uf():
# %%

    state1 = swap23(TensorProduct(Q.S, Q.CU(Q.Y), Q.I))


    # %%
    state2 = state1 * TensorProduct(Q.I, Q.I, Q.Rx(sp.pi/4), Q.I)

    # %%
    state3 = state2 * swap23(TensorProduct(Q.I, Q.CU(Q.Sx), Q.I))

    # %%
    op1 = TensorProduct(Q.I, Q.SWAP)
    op2 = TensorProduct(Q.SWAP, Q.I)
    XCC = op1@op2@Q.CCNOT@op2@op1
    state4 = state3 * swap23(TensorProduct(XCC, Q.I))
    return state4
# %% 
def M_a(a, N):
    dim = 2 ** (int(np.log2(N)) + 1) 
    M = np.zeros((dim,dim), dtype=int)

    for x in range(N):
        xp = (a*x) % N
        M[xp,x] = 1
    for x in range(N, dim):
        M[x,x] = 1
    return sp.Matrix(M.reshape(dim, dim))