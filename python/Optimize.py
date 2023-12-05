import numpy as np
from scipy.optimize import least_squares
import autograd.numpy as np
from autograd import grad, jacobian


def jacobian_autograd(x0):

    def cost(x):
        return x[0]**2 / x[1] - np.log(x[1])

    # gradient_cost = grad(cost)
    jacobian_cost = jacobian(cost)

    # gradient_cost(x)
    J_m=jacobian_cost(x0)
    return J_m

def Least_Squares(fun, x0, jac):
    res1=least_squares()

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
# r, lambda_weight) :

# """Levenberg Marquardt Update
# :param J: Jacobin Matrix J(x) of residual error function r(x),
# : param r: residual error function, dim: (N, n r_out)
# :param labmda_weight: the damping vector, dim: (N, n_r_in)
# : return delta x: update vector, dim: (N, n r_in)
# dim:
# (N,
# n r out,
# n r_in): return delta x norm: norm of
# N = J . shape[Ø] # batch size
# the update vector, dim: (N, 1)"""

    # N= J.shape[2] #bar
    # n f in
    # n_f_out = J. shape[l]
    # # Compute Update Vector :
    # - + \lambda
    # H),
    # dim:
    # JtJ
    # JtR
    # J . transpose(l, 2)
    # torch. bmm(Jt, J)
    # torch. bmm(Jt, r. view(N,
    # (N,
    # n f in,
    # n f out,
    # 1))
    # batched_mat_diag(lambda_weight *
    # - torch. bmm(batched ,
    # delta x =
    # # batch transpose (H, W) to (W,
    # # dim: (N, n_f_in)
    # # dim: (N, 1)
    # JtR) . view(N, n_f_in)
    # dim=l)) . detach()
    # # dim.
    # # dim.
    # # dim.
    # delta_x_norm=
    # return delta_x, delta_x_norm




# Update the
# delta_x, delta_norm =lm_update(J,r,lambda_w)


# # Update parameter
# x= x + delta_x

# sum_delta_norm = torch. max(delta_norm).item()
# if sum_delta_norm < eps:
#     converged_flag=True
#     break
# r=f(x)