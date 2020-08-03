cd  ~/nibbler/matlab/Import/biosig_160
biosig_installer
cd([BCI_DIR 'tasks/conversions']);

data_in= '/home/neuro/data/SFB618/data/eeg/eegRaw/UKBF/';
data_out= '/home/neuro/data/SFB618/data/eeg/eegMat/UKBF/';
dir_list= dir([data_in '00*']);

for di= 1:length(dir_list),
  file= [data_in dir_list(di).name '/data.eeg'];
  [s,hdr]= sload(file);
  tag= 'MontageRaw = ';
  is= strfind(hdr.Header, tag) + length(tag);
  ii= find(hdr.Header==10); 
  ie= ii(min(find(ii>is)))-2;
  cnt.x= s;
  %cnt.clab= hdr.Label;
  cnt.clab= strread(hdr.Header(is:ie),'%s','delimiter',',')';
  cnt.fs= hdr.SampleRate;
  cnt.file= file;
  cnt.title= ['data set ' dir_list(di).name];
  mrk= [];
  mnt= getElectrodePositions(cnt.clab);
  grd= sprintf('F3,Fp1,Fz,Fp2,F4\nT5,C3,Cz,C4,T6\nP3,O1,Pz,O2,P4');
  mnt= mnt_setGrid(mnt, grd);
  
  
  
  %hdr= rmfield(hdr, 'PID');
  %hdr.Patient= rmfield(hdr.Patient, 'Id');
  eegfile_saveMatlab([data_out 'data' dir_list(di).name], cnt, mrk, mnt, ...
                     'format','double', 'vars',{'hdr',hdr});
end

