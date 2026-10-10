import numpy as np, glob
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path="selfie.tflite",num_threads=4); it.allocate_tensors()
i=it.get_input_details()[0]; o=it.get_output_details()[0]
fs=sorted(glob.glob("src/*.png")); prev=None
for p in fs:
    im=Image.open(p).convert("RGB")
    a=(np.asarray(im.resize((256,256),Image.BILINEAR))/255.).astype(np.float32)[None]
    it.set_tensor(i['index'],a); it.invoke(); m=np.clip(it.get_tensor(o['index'])[0,:,:,0],0,1)
    m=m if prev is None else 0.6*m+0.4*prev; prev=m
    Image.fromarray((m*255).astype(np.uint8)).resize(im.size,Image.BICUBIC).save(p.replace("src","mat"))
