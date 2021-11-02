import os

DIR_PATH = os.getcwd()
DATA_PATH = DIR_PATH+"/data2"

# Download data from Figshare
if not (os.path.exists(DATA_PATH)) :
    os.mkdir(DATA_PATH)
    os.chdir(DATA_PATH)
    
    file_num = 1
    
    while True:
        try:
            file_num = int(input("Which file do you want to download? (Select from 1-100) : "))
            break
        except: 
            print("Please input integer")
            
    file_num_str = "00"+str(file_num)
    os.system("wget https://ndownloader.figshare.com/files/16129505 -O cesm{}.tar.gz".format(file_num_str))
    os.system("tar -xf cesm{}.tar.gz".format(file_num_str))
    os.system("rm cesm{}.tar.gz".format(file_num_str))
    os.system("mv member_001 data1")

# Downloading appropriate python libraries
os.chdir(DIR_PATH)
print(os.getcwd())
os.system("pip install -r requirements.txt")
