import glob,os,pickle
os.chdir(R'D:\Github\CarND-Vehicle-Detection-and-Tracking\dataset')

car_list = []
non_car_list = []

os.chdir(R'vehicles')
folders = glob.glob('*')
for j in range(len(folders)):
    if not folders[j] in os.getcwd():
        os.chdir(folders[j])
    car_list += glob.glob('*')
    os.chdir('..')

os.chdir(R'../non-vehicles')
folders = glob.glob("*")
for j in range(len(folders)):
    if not folders[j] in os.getcwd():
        os.chdir(folders[j])
    non_car_list += glob.glob('*')
    os.chdir('..')

len(car_list)
len(non_car_list)
