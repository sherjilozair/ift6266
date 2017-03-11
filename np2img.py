import numpy as np
from PIL import Image
import sys

fname = sys.argv[1]
mb = np.load(fname).squeeze()
n = int(np.sqrt(len(mb)))
mbsz, xdim, ydim = mb.shape[:3]
channel = mb.shape[3] if len(mb.shape) == 4 else 1
mode = 'RGB' if channel == 3 else 'L'
bigimg = Image.new(mode, (xdim*n, ydim*n))

for i, x in enumerate(mb):
    a = i / n
    b = i % n
    img = Image.fromarray(np.clip((x * 255).astype('uint8'), 0, 255))
    bigimg.paste(img, (a*xdim, b*ydim))
fname2 = fname.replace('.npy', '.jpg')
bigimg.save(fname2)
print fname2
