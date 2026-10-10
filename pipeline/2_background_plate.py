import numpy as np, cv2, glob
fs=sorted(glob.glob("src/*.png"))
idx=list(range(0,len(fs),3))
imgs=np.stack([cv2.imread(fs[i]) for i in idx]).astype(np.float32)
ms=[cv2.dilate(cv2.imread(fs[i].replace("src","mat"),0),np.ones((25,25),np.uint8)) for i in idx]
ms=np.stack(ms)>25
imgs[ms]=np.nan
plate=np.nanmedian(imgs,axis=0)
cnt=(~ms).sum(0)
plate[cnt<8]=np.nan
print("never-visible frac",np.isnan(plate[...,0]).mean())
np.save("plate.npy",plate)
cv2.imwrite("plate.png",np.nan_to_num(plate).astype(np.uint8))
