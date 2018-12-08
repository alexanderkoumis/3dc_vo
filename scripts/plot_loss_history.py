#!/usr/bin/env python3

import sys
import json

import matplotlib.pyplot as plt

fname = sys.argv[1]

history = json.loads(open(fname, 'r').read())

for key, vals in history.items():
	plt.plot(vals, label=key)


plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

