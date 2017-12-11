'''
This file is use for training purpose.
In it, the user will going to enter his Name & Student No.
After it, it will capture 20images of the user.
Then the name and student no. is going to save in a seperate csv file

'''

import numpy
import cv2
import os
import dlib
import csv

Dept_Map={10 : "CSE" ,13 :"IT",31: "EC", 00: "CE", 21: "EN", 32: "EI",43: "ME",
          70: "MBA", 14:"MCA"} 

stud_name= input("Enter Name of Student: ")
stud_no = input("Enter the Student No.: ")

Id= stud_no[-3:]
Year = stud_no[0:2]
Branch_Id = int(stud_no[2:4])
Dept = Dept_Map.get(Branch_Id)

if Dept == 'MBA':
    Batch = '20'+Year+'-20'+str(int(Year)+2)
elif Dept == 'MCA':
    Batch = '20'+Year+'-20'+str(int(Year)+3)
else:
    Batch = '20'+Year+'-20'+str(int(Year)+4)


stud_folder_name= "user_"+Id
folderPath=os.path.join(os.path.dirname(os.path.realpath('__file__')),"student\\"+Dept+"\\"+Batch+"\\"+stud_folder_name)

if not os.path.exists(folderPath):
    os.makedirs(folderPath)

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

Num_img = 0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img,1)
    
    for i,d in enumerate(dets):
        Num_img += 1
        cv2.imwrite(folderPath + "/User." + Id + "." + str(Num_img) + ".jpg",
                    img[d.top():d.bottom(),d.left():d.right()])
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
        cv2.waitKey(200)
    cv2.imshow('frame',img)
    cv2.waitKey(1)
    if(Num_img >=20):
        break    
cap.release()
cv2.destroyAllWindows()

path_database = os.path.join(os.getcwd(),"database/")
file_name_csv= path+"stud_info_"+Dept+"_"+Batch+".csv"

if not os.path.exists(file_name_csv):
    with open(file_name_csv,'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',') 
        filewriter.writerows([['Student_No','Name'],
                            [stud_no,stud_name]])
        
        csvfile.close()
flag=0  
with open(file_name_csv) as f:
    csv_file= csv.reader(f,delimiter=',')
    for row in csv_file:
        if stud_no==row[0]:
            flag=1
f.close()

if flag==0:
     with open(file_name_csv,'a',newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',') 
        filewriter.writerow([stud_no,stud_name])
        csvfile.close()
        
        
        
        