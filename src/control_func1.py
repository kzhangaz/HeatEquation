import torch
from math import pi

def control_func(test):
	if test == 0:
		d = {'x' : lambda x: x, 't' : lambda x: x}
		return d
	if test == 1:
		d = {'x' : lambda x: x, 't' : lambda x: torch.sin(pi*x)}
		return d