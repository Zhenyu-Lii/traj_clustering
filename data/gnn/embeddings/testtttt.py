from moviepy.editor import ImageSequenceClip
import numpy as np

fps = 1
filename = "/content/embeddings.gif"
imagefilename = 'images_epoch50.npy'
images = np.load('./'+imagefilename)
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif(filename, fps=fps)
#%%
from IPython.display import Image
with open('/content/embeddings.gif','rb') as f:
    display(Image(data=f.read(), format='png'))