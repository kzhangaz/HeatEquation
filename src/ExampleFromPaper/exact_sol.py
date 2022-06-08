import matplotlib.pyplot as plt
from sympy import *
from math import pi
import numpy as np

def ExactSolution(test,image_path):

	if test == 0:

		# solve ode
		f = Function('f')
		x = symbols('x')
		ode = Eq(f(x).diff(x,2)+f(x), 1)
		ode_sol = dsolve(ode, ics = {f(0):0,f(pi):0})
		sol_func = lambdify(x, ode_sol.rhs)
		# plot solution
		x_1 = np.linspace(0,pi)
		y_1 = sol_func(x_1)
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
		ax1.plot(x_1, y_1, 'r')
		ax1.set_title('Exact solution with u(x) = 1')
		ax1.set_xlabel('x')
		plt.savefig(image_path+'/exactsol_test0.jpg')
		# plot control
		fig2 = plt.figure()
		ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
		ax2.plot(x_1,np.ones(x_1.shape))
		ax2.set_title('control')
		ax2.set_xlabel('x')
		ax2.set_ylabel('u(x)')
		plt.savefig(image_path+'/control_test0.jpg')
		return sol_func

	if test == 1:

		# solve ode
		f = Function('f')
		x = symbols('x')
		ode = Eq(-f(x).diff(x,2)+f(x), x)
		ode_sol = dsolve(ode, ics = {f(0):0,f(pi):0})
		sol_func = lambdify(x, ode_sol.rhs)
		# plot solution
		x_1 = np.linspace(0,pi)
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
		ax1.plot(x_1, sol_func(x_1), 'r')
		ax1.set_title('Solution of the exact problem with u = x')
		ax1.set_xlabel('x')
		ax1.set_ylabel('p(x)')
		plt.savefig(image_path+'/exactsol_test1.jpg')
		# plot control
		fig2 = plt.figure()
		ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
		ax2.plot(x_1,x_1)
		ax2.set_title('control')
		ax2.set_xlabel('x')
		ax2.set_ylabel('u(x)')
		plt.savefig(image_path+'/control_test1.jpg')
		return sol_func

	if test == 2:

		# solve ode
		f = Function('f')
		x = symbols('x')
		ode = Eq(-f(x).diff(x,2)+f(x), sin(pi*x))
		ode_sol = dsolve(ode, ics = {f(0):0,f(pi):0})
		sol_func = lambdify(x, ode_sol.rhs)
		# plot solution
		x_1 = np.linspace(0,pi)
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
		ax1.plot(x_1, sol_func(x_1), 'r')
		ax1.set_title('Solution of the exact problem with u = sin(\pi x)')
		ax1.set_xlabel('x')
		ax1.set_ylabel('p(x)')
		plt.savefig(image_path+'/exactsol_test2.jpg')
		# plot control
		fig2 = plt.figure()
		ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
		ax2.plot(x_1,sin(pi*x))
		ax2.set_title('control')
		ax2.set_xlabel('x')
		ax2.set_ylabel('u(x)')
		plt.savefig(image_path+'/control_test2.jpg')
		return sol_func

	if test == 3:

		# solve ode
		f = Function('f')
		x = symbols('x')
		ode = Eq(-f(x).diff(x,2)+f(x), sin(8*x))
		ode_sol = dsolve(ode, ics = {f(0):0,f(pi):0})
		sol_func = lambdify(x, ode_sol.rhs)
		# plot solution
		x_1 = np.linspace(0,pi)
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
		ax1.plot(x_1, sol_func(x_1), 'r')
		ax1.set_title('Solution of the exact problem with u = sin(8x)')
		ax1.set_xlabel('x')
		ax1.set_ylabel('p(x)')
		plt.savefig(image_path+'/exactsol_test3.jpg')
		# plot control
		fig2 = plt.figure()
		ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
		ax2.plot(x_1,sin(8*x))
		ax2.set_title('control')
		ax2.set_xlabel('x')
		ax2.set_ylabel('u(x)')
		plt.savefig(image_path+'/control_test3.jpg')
		return sol_func

	if test == 4:

		# solve ode
		f = Function('f')
		x = symbols('x')
		ode = Eq(-f(x).diff(x,2)+f(x), np.exp(-80*(x-pi/2)^2))
		ode_sol = dsolve(ode, ics = {f(0):0,f(pi):0})
		sol_func = lambdify(x, ode_sol.rhs)
		# plot solution
		x_1 = np.linspace(0,pi)
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
		ax1.plot(x_1, sol_func(x_1), 'r')
		ax1.set_title('Solution of the exact problem with u = np.exp(-80*(x-pi/2)^2)')
		ax1.set_xlabel('x')
		ax1.set_ylabel('p(x)')
		plt.savefig(image_path+'/exactsol_test4.jpg')
		# plot control
		fig2 = plt.figure()
		ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
		ax2.plot(x_1,np.exp(-80*(x-pi/2)^2))
		ax2.set_title('control')
		ax2.set_xlabel('x')
		ax2.set_ylabel('u(x)')
		plt.savefig(image_path+'/control_test4.jpg')
		return sol_func

