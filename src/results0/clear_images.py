import os
import glob

files = glob.glob('src/results/simulation/*.jpg')

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

files = glob.glob('src/results/estimation/*.jpg')

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))