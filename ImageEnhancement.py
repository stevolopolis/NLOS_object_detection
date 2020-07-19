import cv2 
import os 

obj_img = cv2.imread(r"C:\Users\tangk\Pictures\CanonEOS\Day2\Reference Frames\2020_07_19_IMG_0155.JPG",1)
ref_img = cv2.imread(r"C:\Users\tangk\Pictures\CanonEOS\Day2\Reference Frames\2020_07_19_IMG_0156.JPG",1)

crop_dim = (500,500)
obj_img = cv2.resize(obj_img,crop_dim)
ref_img = cv2.resize(ref_img,crop_dim)

obj_img = cv2.GaussianBlur(obj_img,(5,5),0)
ref_img = cv2.GaussianBlur(ref_img,(5,5),0)

cv2.imshow('img',ref_img)
cv2.waitKey(0)   
cv2.destroyAllWindows()

cv2.imshow('img',obj_img)
cv2.waitKey(0)    
cv2.destroyAllWindows()

diff_img = cv2.subtract(obj_img,ref_img)

cv2.imshow('img',diff_img)
cv2.waitKey(0)  
cv2.destroyAllWindows()

amp_img = diff_img*20
cv2.imshow('img',amp_img) 
cv2.waitKey(0)   
cv2.destroyAllWindows()
