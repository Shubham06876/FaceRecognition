# This is to test our model. We first need to train model before testing it.
# Run this module after running ModelTrainer.py


import cv2
import os
import faceRecognition as fr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "TestImages")


# This is testing module for face recognition system.

# subject Image path
# path = os.path.join(image_dir, "Kangana.jpg")
# path = os.path.join(image_dir, "Priyanka1.jpg")
# path = os.path.join(image_dir, "Priyanka2.jpg")
path = os.path.join(image_dir, "Priyanka3.jpg")



subjectImage=cv2.imread(path)
# conversion of color image to gray image increases detect rates.
faces_detected_in_image,gray_img=fr.detect_faces(subjectImage)
print("faces detected:",faces_detected_in_image)



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
# we load TrainedModel.yml which has our trained model stored.
face_recognizer.read('TrainedModel.yml')

label={0:"Priyanka",1:"Kangana"}  # creating dictionary containing labels for each identity

for face in faces_detected_in_image:
    (x,y,w,h)=face
    # x is left co-ordinate of image, x+w is right, y is bottom, y+h is top
    roi_gray = gray_img[y:y+h,x:x+w]
    # predicting the identity of given image and uncertainity of prediction
    # lower the uncertainity value, higher the confidence and less are chances of match.
    identity,uncertainity=face_recognizer.predict(roi_gray)
    # print("uncertainity:",uncertainity)
    print("identity:",identity)
    fr.draw_rect(subjectImage,face)
    predicted_label=label[identity]

    # If uncertainity more than 60 then don't print predicted face text on screen
    # This is to ensure that our model doesnt wrongly identity any face.
    if(uncertainity<65):
        fr.label_face(subjectImage, predicted_label, x, y)

display_image=cv2.resize(subjectImage,(1000,1000))
cv2.imshow("Face Recognition System",display_image)
cv2.waitKey(0) # wait until some key is pressed.
cv2.destroyAllWindows





