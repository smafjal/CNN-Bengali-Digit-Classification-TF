#!usr/bin/env python
__author__="smafjal"

from PIL import Image
import os
import glob
from skimage.io import imread
import numpy as np
import pickle

class ImageReader():
    def __init__(self, dir):
        self.data_dir = dir

    def read_image(self):
        files = os.listdir(self.data_dir)
        image_list = []
        label_list=[]
        for filename in files:
            label = self.get_label(filename)
            if label == -1:
                print "Label is not found"
                continue

            filename = os.path.join(self.data_dir, filename) + "/*.bmp"
            for img_file in glob.glob(filename):
                img=Image.open(img_file).convert("L")
                img=np.array(img).reshape(-1)
                image_list.append(img)
                a=[0]*10; a[label]=1 # making hot vector
                label_list.append(a)
        return np.array(image_list),np.array(label_list)

    def get_label(self, filename):
        for i in range(10):
            if int(filename[-1]) == i:
                return i
        return -1

    def save_pickle(self,data,file_name):
        with open(file_name+".pickle","wb") as w:
            pickle.dump(data,w)

def main():
    data = ImageReader("bengali_digit")
    img_data,img_label = data.read_image()
    data.save_pickle(img_data,"img_data")
    data.save_pickle(img_label,"img_label")

    print "Data-Len: ",len(img_data)


if __name__ == "__main__":
    main()
