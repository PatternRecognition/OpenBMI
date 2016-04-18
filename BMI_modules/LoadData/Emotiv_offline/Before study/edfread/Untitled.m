clc 
clear all;

filename='123-5555-19.02.16.19.51.19.edf';

[dat, marker, hdr]=Load_Emotiv(filename);


% 
% [Temp_hdr Temp_data]=edfread(filename);



% 
% hdr=Load_EM_hdr(filename); disp('Loading EEG header file..');
% data=Load_EM_data(filename, hdr); disp('Loading EEG header file..');
% marker=Load_EM_marker(filename, hdr); disp('Loading EEG header file..');
% 





% hdr=Load_EM_hdr
% 
% dat=load_EM_data
% 
% marker-Load_EM_mrk