import cv2
import os


data_path = 'data/'
# first, get the list of the images located in the data_path folder. These images names (e.g. 'tracking_0001.jpeg') will
# be used for indexing.
trackingImages = [name for name in os.listdir(os.path.join(os.curdir, "data/")) if not name.startswith('.')]
# We sort this list to get the names in increasing order
trackingImages.sort(key=lambda s: s[10:13])
trackingImages.sort()
images = []
# loop over the image paths
for imagePath in trackingImages:
    imagePath = data_path + imagePath
    # load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)

i = 0
xoff = 220
yoff = 400
print('{}'.format(len(images)))

while i < len(images):
    i += 1
    curr_img = images[i]
    if i < len(images)/4:
        sig = curr_img[yoff, xoff:600]

        print('Image number : {} ; {}'.format(i, min(sig)))
        cv2.rectangle(curr_img, (xoff, yoff), (xoff+(600-xoff), yoff), (0, 0, 0), 2)
        #cv2.rectangle(curr_img)
    cv2.imshow("Tracking", curr_img)
    cv2.waitKey(0)



