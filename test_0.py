import warnings
import scipy

import numpy as np

from numpy.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_array_almost_equal

from scipy.sparse import csr_matrix, coo_matrix, SparseEfficiencyWarning

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation, \
    classical_interpolation
    

# print("################Test 1 basic################")
# # Test 1
# import pyamg
# import numpy as np
# A = pyamg.gallery.poisson((100,100), format='csr')  # 2D Poisson problem on 500x500 grid
# ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
# print(ml)                                           # print hierarchy information
# b = np.random.rand(A.shape[0])                      # pick a random right hand side
# x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
# print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector




# # Test 2(e2e)
# print("################ Test 2(e2e)################")
# from numpy import ones
# from pyamg import ruge_stuben_solver
# from pyamg.gallery import poisson

# A1 = poisson((100, 100), format='csr')  # 2D Poisson problem on 100x100 grid(convert from grid to matrix)
# # Usually computers are not contuniuous, maths is, so discretize the problem.
# b1 = np.random.rand(A1.shape[0])
# MultiLevelSolver_obj = ruge_stuben_solver(A1, max_coarse=20, max_levels=20, coarse_solver='cholesky')     # setup phase
# print(MultiLevelSolver_obj)      
# residuals1 = []
# x0_initial_guess  = np.ones(A1.shape[0])     # initial guess
# x1 = ml.solve(b = b1, x0 = x0_initial_guess, tol=1e-12, residuals=residuals1, cycle='V', maxiter=200, accel='gmres') # standalone solver
# print("residual: ", np.linalg.norm(b1 - A1 @ x1))


# # Test 3(Multilevel)
# print("################ Test 3(MultilevelSolver class)################")
# print("Called from - ruge_stuben_solver")
# print("Performs setup, V cycle, and calls coarse solver")
# print("######################")
# # manual construction of a two-level AMG hierarchy
# from pyamg.gallery import poisson
# from pyamg.multilevel import MultilevelSolver
# from pyamg.strength import classical_strength_of_connection
# from pyamg.classical.interpolate import direct_interpolation
# from pyamg.classical.split import RS

# # compute necessary operators
# A = poisson((100, 100), format='csr')
# C = classical_strength_of_connection(A)
# splitting = RS(A) # RS = Ruge-Stuben splitting, The most simplest splitting technique, C/F splitting...
# P = direct_interpolation(A, C, splitting)
# R = P.T

# # store first level data
# levels = []
# levels.append(MultilevelSolver.Level())
# levels.append(MultilevelSolver.Level())
# levels[0].A = A
# levels[0].C = C
# levels[0].splitting = splitting
# levels[0].P = P
# levels[0].R = R

# # store second level data
# levels[1].A = R @ A @ P  # coarse-level matrix
# # No R and P as it's unnecessary for the coarsest level

# # create MultilevelSolver
# # levels is vector of = [{R1, A1, P1, splitting1, C1}, {R2, A2, P2, splitting2, C2}, ..., ...]
# ml = MultilevelSolver(levels, coarse_solver='cg')
# print(ml)


# # Test 4 (e2e, using poisson eqn).
# # (smoothed aggregation solver - A modified AMGq)
# print("################ Test 4 (e2e, using poisson eqn.), use smoothed aggregation solver################")
# import scipy
# import numpy
# import pyamg

# stencil = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
# A3 = pyamg.gallery.stencil_grid(stencil, (100, 100), dtype=float, format='csr')

# B3 = numpy.ones((A3.shape[0], 1))
# multilevel_obj3 = pyamg.smoothed_aggregation_solver(A3, B3, max_coarse=10)

# print(f'Number of nonzeros: {A3.nnz} \n'
#       f'Shape of the matrix: {A3.shape[0]} x {A3.shape[1]} \n'
#       f'Format of the matrix: {A3.format}\n')
# print(multilevel_obj3)

# residuals = []
# b3 = numpy.random.rand(A3.shape[0], 1)
# x0_3 = numpy.random.rand(A3.shape[0], 1)

# x3 = multilevel_obj3.solve(b=b3, x0=x0_3, tol=1e-10, residuals=residuals)
# print(f'Residual: {numpy.linalg.norm(b3 - A3 @ x3)}')


# # Test 5 (construct sparse matrix from a grid.)
# print("################ Test 5 (construct sparse matrix from a grid.)################")
# from pyamg.gallery import stencil_grid
# stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]] # 2D Poisson stencil
# grid = (3,3)                            # 2D grid with shape 3x3
# Matrix = stencil_grid(stencil, grid, dtype=float, format='csr')
# print("Stencil: ", stencil)
# print("Grid: ", grid)
# print(f"Sparse Matrix({Matrix.format}): \n", Matrix.toarray())


# # Test 6 (Only coarse grid solver)
# print("################ Test 6 (Only coarse grid solver)################")
# print("Running different course grid solvers only")


# from pyamg import coarse_grid_solver
# from scipy.sparse.linalg import bicg
# A4 = poisson((10, 10), format='csr')     # take a stencil convert to a sparse matrix in csr.
# b4 = A4 @ np.ones(A4.shape[0])

# print("Dense solver: LU factorization")
# cgs = coarse_grid_solver('lu')
# x4_lu = cgs(A4, b4)

# print("Spare direct method solver: splu")
# splu = coarse_grid_solver('splu')
# x4_splu = splu(A4, b4)

# print("kryvlov solver: cg")
# cg_solver = coarse_grid_solver('cg')
# x4_cg = cg_solver(A4, b4)

# print("scipy.sparse.linalg solver: bicg")
# x4_bicg, info = scipy.sparse.linalg.bicg(A4, b4, atol=1e-10) # used atol and not tol.

# # check if all values of x are same?
# print("All values of x are same: ", np.allclose(x4_lu, x4_splu, x4_cg, x4_bicg))
# # check relative error
# print("Relative error: ", np.linalg.norm(x4_lu - x4_splu, np.inf))
# print("Relative error: ", np.linalg.norm(x4_lu - x4_cg, np.inf))
# print("Relative error: ", np.linalg.norm(x4_lu - x4_bicg, np.inf))




# # Test7 (AMG as preconditioner)
# print("################ Test 7 (AMG as preconditioner)################")
# print(" Using as preconditioner = Pre*A*x = Pre*b, Where Pre = preconditioner)")
# from pyamg.aggregation import smoothed_aggregation_solver
# from pyamg.gallery import poisson
# from scipy.sparse.linalg import cg
# import scipy as sp

# A = poisson((100, 100), format='csr')          # matrix
# b = np.random.rand(A.shape[0])                 # random RHS

# ml = smoothed_aggregation_solver(A)            # AMG solver
# M = ml.aspreconditioner(cycle='V')             # preconditioner
# x, info = cg(A, b, atol=1e-8, maxiter=30, M=M)  # solve with CG

# print("X values as precond: ", x)

# x_cg, info_cg = cg(A, b, atol=1e-8, maxiter=30)  # solve with CG
# print("X values without precond: ", x_cg)


# solver testing independetly

# 1. Solve -> solve - no brainer. use any solver, the solve phase matters.



def test_rs_baseline(A, x, b):

      # 2. config
      ruge_stuben_config = {
            'strength': ('classical', {'theta': 0.25}),  # Method to determine connection strength
            'CF': ('RS', {'second_pass': False}),  # Coarse grid selection method
            'interpolation': 'classical',  # Interpolation method
            'presmoother': ('gauss_seidel', {'sweep': 'symmetric'}),  # Presmoother method
            'postsmoother': ('gauss_seidel', {'sweep': 'symmetric'}),  # Postsmoother method
            'max_levels': 30,  # Maximum levels
            'max_coarse': 10,  # Maximum number of variables on coarse grid
            'keep': False,  # Flag to keep strength in hierarchy for diagnostics
            'coarse_solver': 'cg',  # Coarse solver method (default)
      }

      # 3. solver
      ml = ruge_stuben_solver(A, **ruge_stuben_config)
      # 4. solve
      residual = []
      x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12,
                              residuals=residual)
      # 5. assert
      r = b - A*x_sol
      rho = np.dot(r, r)
      return x_sol, rho

def test_cg_scipy(A, x, b):

      x_sol, _ = scipy.sparse.linalg.cg(A, b, x0=x, rtol=1e-12, maxiter=20)

      r = b - A*x_sol
      rho = np.dot(r, r)
      return x_sol, rho

      
case = (250, 250)    #2d
# 1. input 
A = poisson(case, format='csr')
np.random.seed(0)  # make tests repeatable
x = np.random.rand(A.shape[0])
b = A*np.random.rand(A.shape[0])  # zeros_like(x)

x1, r1 = test_rs_baseline(A, x, b)
x2, r2 = test_cg_scipy(A, x, b)

print("rho1: ", r1)
print("rho2: ", r2)
# assert testing
np.testing.assert_allclose(x1, x2, rtol=1e-5)
print("equal")

