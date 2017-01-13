#!usr/bin/env python
__author__="smafjal"

import numpy as np
import pickle

def read_data(data_path):
    with open(data_path,"r") as f:
        data=pickle.load(f)
    return data

def main():
    data_x=read_data("data_dir/img_data.pickle")
    data_y=read_data("data_dir/img_label.pickle")

    print "Data-Lan: ",len(data_x)

if __name__=="__main__":
    main()
