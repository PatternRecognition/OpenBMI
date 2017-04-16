function [dat,marker, hdr, HDR]=Load_Emotiv(filename)

[Temp_hdr Temp_data]=edfread(filename);

[hdr, dat ,marker]=Convert_Emotiv(Temp_hdr,Temp_data);



% hdr.Label=Temp_hdr.label';
% 
% hdr.NS=length(hdr.Label);
% 
% [HDR] = leadidcodexyz(hdr)






