import numpy as np
from PIL import Image
import sys

fname = sys.argv[1]
mb = np.load(fname)
n = int(np.sqrt(len(mb)))

bigimg = Image.new('L', (28*n, 28*n))

for i, x in enumerate(mb):
    x = i / n
    y = i % n
    img = Image.fromarray((x * 255).astype('uint8'))
    bigimg.paste(img, (x*28, y*28))

bigimg.save(fname.replace('.npy', '.jpg'))
