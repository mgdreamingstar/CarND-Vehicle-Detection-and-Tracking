import glob,os,pickle,cv2
import numpy as np

os.chdir(R'D:\Github\CarND-Vehicle-Detection-and-Tracking\dataset')

# 获得图片的文件名 并 获取图片本身，存放在 numpy.array 中
car_list = []
non_car_list = []
car_imgs = np.array([])
non_car_imgs = np.array([])

os.chdir(R'vehicles')
folders = glob.glob('*')
for j in range(len(folders)):
    if not folders[j] in os.getcwd():
        os.chdir(folders[j])
    car_list += glob.glob('*.png')
    for i in range(len(glob.glob('*.png'))):
        if i == 0 and car_imgs.shape[0] == 0:
            car_imgs = cv2.imread(glob.glob('*.png')[i])[None,:]
        else:
            image = cv2.imread(glob.glob('*.png')[i])[None,:]
            car_imgs = np.concatenate((car_imgs, image))
        print(car_imgs.shape[0], i ,(glob.glob('*.png')[i]))
    os.chdir('..')

os.chdir(R'../non-vehicles')
folders = glob.glob("*")
for j in range(len(folders)):
    if not folders[j] in os.getcwd():
        os.chdir(folders[j])
    non_car_list += glob.glob('*')
    for i in range(len(glob.glob('*.png'))):
        if i == 0 and non_car_imgs.shape[0] == 0:
            non_car_imgs = cv2.imread(glob.glob('*.png')[i])[None,:]
        else:
            image = cv2.imread(glob.glob('*.png')[i])[None,:]
            non_car_imgs = np.concatenate((non_car_imgs, image))
        print(non_car_imgs.shape[0], i ,(glob.glob('*.png')[i]))
    os.chdir('..')

len(car_list)
len(non_car_list)