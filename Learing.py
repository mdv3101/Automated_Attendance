# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:01:00 2017

@author: Madhav
"""

import os
import numpy as np
import cv2
import dlib

path = 'D:/DataScience/Attendance/'
opencv_path = "C:/opencv/build/etc/lbpcascades/"
detector = dlib.get_frontal_face_detector()

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x,y), (w,h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def count_face(img):
    dets = detector(img,1)
    return (len(dets))

def detect_face(img):
    dets = detector(img,1)
    for i,d in enumerate(dets):
        #print(i)
        face= img[d.top():d.bottom(),d.left():d.right()]
        top,bot,left,rgt = d.top(),d.bottom(),d.left(),d.right()
        rect1 = left,bot,rgt,top
        #rect1 = d.top(),d.bottom(),d.left(),d.right()
        if (len(face) == 0):
            print ("Unable to detect image")
            return None,None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        return gray, rect1
    
        
 
def prepare_learning_data(dept_name,batch):
    faces=[]
    labels=[]
    learn_path_dir = os.path.join(os.getcwd(),"student\\"+dept_name+"\\" +batch +"\\")
    dirs= os.listdir(learn_path_dir)
    for dir_name in dirs:
        if not dir_name.startswith("user_"):
            continue;
        label = int(dir_name.replace("user_", ""))
        subject_dir_path = learn_path_dir + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            cv2.namedWindow('Training on image...',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Training on image...', 600,600)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            #print(image_name)
            no_of_face_test = count_face(image)
            #print(no_of_face_test) 
            if no_of_face_test ==0:
                print ("Unable to detect image")
                print(image_name)
                face = None
            else:
                face,rect = detect_face(image)
            #if face is None:
            #   print ("Invalid Image in prep")
            if face is not None:
                faces.append(face)
                labels.append(label)
                cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels
    
print("Preparing data...")    
faces, labels = prepare_learning_data("IT","2014-2018")
print("Data Prepared...")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))


    
def predict(test_img):
    img = test_img.copy()
    no_of_face_test = count_face(img)
    #print(no_of_face_test)
    
    if no_of_face_test ==0:
        print ("Unable to detect image")
        return 0
    for i in range(0,no_of_face_test):
        face, rect= detect_face(img)
        #print (face)
        #top,bot,left,rgt=rect
        if img is None:
            print ("Invalid Image")
            return 0
        if face is None:
            print ("Unable to detect image")
            return 0
        label= face_recognizer.predict(face)
        label2 = str(label[0])
        print(label)
        #dlib.rectangle(left,top,rgt,bot)
        draw_rectangle(img, rect)
        draw_text(img, label2, rect[2]-50, rect[3]-10)
    return img
    
test_img1 = cv2.imread(path+ "test-data/test15.jpg")

if test_img1 is not None:
    predicted_img1 = predict(test_img1)
    print("Prediction complete")
    if str(type(predicted_img1)) != "<class 'int'>":  
        cv2.namedWindow('Test_output',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Test_output', 600,600)
        cv2.imshow("Test_output", predicted_img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


