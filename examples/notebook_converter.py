

import IPython.nbformat.current as nbf
nb = nbf.read(open('lane_finder.py', 'r'), 'py')
nbf.write(nb, open('lane_finder.ipynb', 'w'), 'ipynb')