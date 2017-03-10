import numpy as np
from PIL import Image
import sys

fname = sys.argv[1]
mb = np.load(fname).squeeze()
n = int(np.sqrt(len(mb)))
bigimg = Image.new('L', (28*n, 28*n))

for i, x in enumerate(mb):
    a = i / n
    b = i % n
    img = Image.fromarray(np.clip((x * 255).astype('uint8'), 0, 255))
    bigimg.paste(img, (a*28, b*28))
fname2 = fname.replace('.npy', '.jpg')
bigimg.save(fname2)
print fname2
