data_in= '/home/neuro/data/SFB618/data/eeg/eegRaw/UKBF/';
data_out= '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
dir_list= dir([data_in '00*']);

for di= 2:length(dir_list),
  [cnt, mrk, mnt, hdr] = eegfile_loadMatlab([data_out 'data' dir_list(di).name], 'vars',{'dat','mrk','mnt','hdr'}) ;
  mrk = getMarkerFromHeader(hdr,cnt.fs) 
  eegfile_saveMatlab([data_out 'data' dir_list(di).name], cnt, mrk, mnt, 'format','double', 'vars',{'hdr',hdr});
end

