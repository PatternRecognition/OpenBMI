clear all; close all; clc;
a=zeros(1,18775);
predicted=py.SharedDemo_5_Online.decoding(a)
data=double(py.array.array('d',py.numpy.nditer(predicted)))
