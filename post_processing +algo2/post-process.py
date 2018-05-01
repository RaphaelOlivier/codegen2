import sys
import os
import numpy as np

file1 = sys.argv[1]

def post_process_HS():

	fileptr = open(file1, 'r')
	for instance in fileptr:
		print (instance)

if __name__=="__main__":
    post_process_HS()