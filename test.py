import skimage
import skimage.io
import skimage.transform


# img = skimage.io.imread('static/img/face.jpeg')  # load image from file
img = skimage.io.imread('static/img/bfcngT8S8Uc.jpg')  # load image from file
shape = list(img.shape)
print(shape)
if shape[0] < shape[1]:
    size = shape[0]
    x = (shape[1] - size) / 2
    img = img[0:size, x:x + size]
else:
    size = shape[1]
    img = img[0:size, 0:size]

img = skimage.transform.resize(img, (224, 224, 3))
skimage.io.imsave('static/img/res.jpeg', img)

print(img.shape)
# skimage.transform.resize
