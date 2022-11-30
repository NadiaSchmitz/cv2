import cv2
import numpy as np

image = cv2.imread('images/my_image.jpg')
#print(image.shape)

image = cv2.resize(image, (300, 200))

# image[0:100, 0:100]
#cv2.imshow('Belochka', image)
#cv2.waitKey(0)

new_image = cv2.GaussianBlur(image, (11, 11), 0, 0)
#cv2.imshow('Belochka', new_image)
#cv2.waitKey(0)

new_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Belochka', new_image_1)
#cv2.waitKey(0)

new_image_2 =cv2.Canny(new_image_1, 200, 200)
#cv2.imshow('Belochka', new_image_2)
#cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)

new_image_3 = cv2.dilate(new_image_2, kernel, iterations=1)
#cv2.imshow('Belochka', new_image_3)
#cv2.waitKey(0)

new_image_4 = cv2.erode(new_image_3, kernel, iterations=1)
#cv2.imshow('Belochka', new_image_4)
#cv2.waitKey(0)

#capture = cv2.VideoCapture('videos/my_video.mp4')
#capture.set(3, 500)
#capture.set(4, 300)


#while True:
    #success, img = capture.read()
    #cv2.imshow('Kimono', img)

    #if cv2.waitKey(10) & 0xFF == ord('q'):
        #break

# 3

#photo = np.zeros((450, 450, 3), dtype='uint8')

#photo[100:150, 200:280] = 85, 94, 215

#cv2.rectangle(photo, (10, 10), (100, 100), (85, 94, 215), thickness=3)
#cv2.rectangle(photo, (100, 100), (200, 200), (85, 94, 215), thickness=cv2.FILLED)
#cv2.line(photo, (0, 200), (100, 200), (85, 94, 215))
#cv2.line(photo, (5, photo.shape[0] // 3), (100, photo.shape[1] // 2), (85, 94, 215))

#cv2.imshow('New_Photo', photo)
#cv2.waitKey(0)

# Floor
capture = cv2.VideoCapture('videos/floor.mp4')
dataset = []
n = 5000
i=0
while True:
    success, img = capture.read()
    try:
        new_new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        break
    new_new_new_img = cv2.Canny(new_new_img, 200, 200)
    kernel = np.ones((5, 5))
    new_new_new_new_img = cv2.dilate(new_new_new_img, kernel, iterations=1)
    new_new_new_new_img = cv2.erode(new_new_new_new_img, kernel, iterations=1)

    dataset.append(new_new_new_new_img)
    cv2.imshow('Floor', new_new_new_new_img)
    path = r'C:\Users\DAA\github\ai_image_video\videos\dataset\img_' + str(i) + '.jpg'
    cv2.imwrite(path, new_new_new_new_img)
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

print("Anzahl der Bilder: ", len(dataset))
