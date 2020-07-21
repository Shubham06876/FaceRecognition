import cv2
import os
import numpy as np

# This module contains all common functions that are called in TestModel.py file


# Given an image below function returns rectangle for face detected alongwith gray scale image
def detect_faces(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    # Load haar classifier, selected from cv2 module
    haar_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
    detected_faces = haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)

    return detected_faces,gray_img


#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer


# Print label for face detected.
def label_face(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)

# draw rectangle around face
def draw_rect(test_img,face):
    (x,y,w,h)=face

    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)  # BGR combination

# Get label for input image
def get_labels_for_images(directory):
    faces=[]
    faceID=[]

    for dir,subdir,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")#Skipping hidden files
                continue

            id=os.path.basename(dir)#fetching subdirectory names
            img_path=os.path.join(dir,filename)#fetching image path
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)#loading each image one by one
            if test_img is None:
                print("Error while loading image")
                continue
            faces_rect,gray_img=detect_faces(test_img)#Calling detect_faces function to return faces detected in particular image
            # to correctly train our model images with only one face should be used in training
            if len(faces_rect)!=1:
               continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID














