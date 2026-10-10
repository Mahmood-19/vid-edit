import numpy as np, cv2, glob, os
os.makedirs("mat2",exist_ok=True)
plate=np.load("plate.npy"); valid=~np.isnan(plate[...,0]); plate=np.nan_to_num(plate)
fs=sorted(glob.glob("src/*.png"))
for p in fs:
    f=cv2.imread(p).astype(np.float32)
    m=cv2.imread(p.replace("src","mat"),0).astype(np.float32)/255
    sm=np.clip((m-0.42)/0.28,0,1)
    fb=cv2.GaussianBlur(f,(0,0),1.2); pb=cv2.GaussianBlur(plate,(0,0),1.2)
    d=np.abs(fb-pb).max(2)
    fg=np.clip((d-22)/16,0,1)*valid
    fg=cv2.morphologyEx((fg*255).astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8)).astype(np.float32)/255
    hsv=cv2.cvtColor(f.astype(np.uint8),cv2.COLOR_BGR2HSV)
    skin=((hsv[...,1]>45)&((hsv[...,0]<25)|(hsv[...,0]>168))).astype(np.uint8)
    skin=cv2.dilate(cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)),np.ones((13,13),np.uint8)).astype(np.float32)
    fg=fg*skin
    gate=cv2.dilate((sm>0.5).astype(np.uint8),np.ones((61,61),np.uint8)).astype(np.float32)
    # only keep diff blobs connected to the person
    n,lab,st,_=cv2.connectedComponentsWithStats(((fg>0.5)|(sm>0.5)).astype(np.uint8),connectivity=8)
    big=(sm>0.5).astype(np.uint8); keep=np.zeros(n,bool)
    for k in np.unique(lab[big>0]): keep[k]=True
    conn=keep[lab].astype(np.float32)
    a=np.maximum(sm,fg*gate*conn)
    a8=(a*255).astype(np.uint8)
    a8=cv2.morphologyEx(a8,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
    pad=cv2.copyMakeBorder((a8>128).astype(np.uint8),1,1,1,1,cv2.BORDER_REPLICATE)
    cs,_=cv2.findContours(pad,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    fill=np.zeros_like(pad); cv2.drawContours(fill,cs,-1,1,-1); fill=fill[1:-1,1:-1]
    a8=np.maximum(a8,fill*255)
    a=a8.astype(np.float32)/255
    a=cv2.GaussianBlur(a,(0,0),1.0)
    cv2.imwrite(p.replace("src","mat2"),(np.clip(a,0,1)*255).astype(np.uint8))
