clc
clear all;

filename='123-5555-19.02.16.19.51.19.edf';


EEG

[hdr data]=edfread(filename);

EEG=Convert_Emotiv(hdr,data);

 
% 	HDR = sopen(filename, 'r', 0,'OVERFLOWDETECTION:OFF');
% 	[S,HDR] = sread(HDR, NoR, StartPos);
% 	HDR = sclose(HDR);
%     
%     filename=('F:\SMC2015\Ver2.emotiv\yjkee-123-19.02.16.19.36.31.edf');

