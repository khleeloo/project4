import numpy as np
from scipy.optimize import least_squares
from sympy import sin, cos, Matrix
from sympy.abc import rho, phi


def least_squares(error, x0, jac):
    ##Method levenberg-marquardt
    res1=least_squares(error, p1.ravel(), args=(p, kp1, kp2), method='lm')
    return res1

def jac(x0):
    input=x0
    x=input[0]
    y=input[1]
    z=input[2]
    J_m=np.asarray([[[2*x[0]+x[1],x[0]],[3*x[1]*x[1],1+6*x[0]*x[1]]],
    [[2*y[0]+y[1],y[0]],[3*y[1],2*y[1]+3*y[0]]],
    [[z[1],z[0]],[2*z[0],1.0]]],dtype=np.float32)

    return J_m


# def lm_update(J,r,lambda_w)
# # r, lambda_weight) :

# # """Levenberg Marquardt Update
# # :param J: Jacobin Matrix J(x) of residual error function r(x),
# # : param r: residual error function, dim: (N, n r_out)
# # :param labmda_weight: the damping vector, dim: (N, n_r_in)
# # : return delta x: update vector, dim: (N, n r_in)
# # dim:
# # (N,
# # n r out,
# # n r_in): return delta x norm: norm of
# # N = J . shape[Ø] # batch size
# # the update vector, dim: (N, 1)"""

#     N= J.shape[2] #batch update
#     n_f_in=J.shape[2]
#     n_f_out=J.shape[1]

#     Jt=J.transpose(1,2)
#     JtJ=torch.bmm(Jt,J)
#     JtR=torch.bmm(Jt,r.view(N,n_f_out,1))
   
#     return delta_x, delta_x_norm




# Update the
# delta_x, delta_norm =lm_update(J,r,lambda_w)


# # Update parameter
# x= x + delta_x

# sum_delta_norm = torch. max(delta_norm).item()
# if sum_delta_norm < eps:
#     converged_flag=True
#     break
# r=f(x)