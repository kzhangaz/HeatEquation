import torch
from math import pi

def control_func(test):
	if test == 0:
		return lambda x: torch.pow(x,0)
	if test == 1:
		return lambda x: x
	if test == 2:
		return lambda x: torch.sin(pi*x)
	if test == 3:
		return lambda x: torch.sin(8*x)
	if test == 4:
		return lambda x: torch.exp(-80*(x-pi/2)**2)