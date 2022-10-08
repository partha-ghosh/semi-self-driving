import sys
import os

with open(sys.argv[1], 'r') as f:
    s = f.read()

if 'RuntimeError:' in s:
    os.system(sys.argv[2])
    os.system(f'rm -rf `dirname {sys.argv[1]}`')
