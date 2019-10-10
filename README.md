# face_recognition 

## Usage


```
# For each pic in dir: instantiate, load, embed and anchor.

image_dir = ('./image_dir')
paths = os.listdir(image_dir)

pictures  = []

for k,path in enumerate(paths):
    pictures.append(Pic(path))
    pictures[-1].load()    
    pictures[-1].embed()
    pictures[-1].anchor()
```
Output:

```
PictureA            PictureB            VectorDistance
dt1.jpg             dt1.jpg                    0.0
dt1.jpg             kk1.jpg             0.7849338793765491
dt1.jpg             kk2.jpg             0.8149979452891201
kk1.jpg             dt1.jpg             0.7849338793765491
kk1.jpg             kk1.jpg                    0.0
kk1.jpg             kk2.jpg             0.3400226363607682
kk2.jpg             dt1.jpg             0.8149979452891201
kk2.jpg             kk1.jpg             0.3400226363607682
kk2.jpg             kk2.jpg                    0.0

```

```
# Plot facial features

fig, axes = plt.subplots(num,figsize=(4,4*num))

for i in range(num):
    pictures[i].plot(axes[i])
    
```

![faces](https://github.com/adilkhan49/face_recognition/blob/master/faces.png)



