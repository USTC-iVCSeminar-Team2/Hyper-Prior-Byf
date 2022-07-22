from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\EsakaK\Desktop\1.png")
print(np.array(img)[:,:,3])