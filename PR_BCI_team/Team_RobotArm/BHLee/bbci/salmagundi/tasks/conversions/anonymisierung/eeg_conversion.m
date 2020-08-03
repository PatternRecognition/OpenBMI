function [cnt,mrk,mnt] = eeg_conversion(file);

% load the data
try
  hdr = ukbfopen(file);
  s = ukbfread(hdr);
catch
  error('can not load data');
end
  
% extract cnt mrk mnt
tag= 'MontageRaw = ';
is= strfind(hdr.Header, tag) + length(tag);
ii= find(hdr.Header==10); 
ie= ii(min(find(ii>is)))-2;
cnt.x= s;
cnt.clab= strread(hdr.Header(is:ie),'%s','delimiter',',')';
cnt.fs= hdr.SampleRate;
cnt.file= file;
cnt.title= ['data set ' file];
mrk= getMarkerFromHeader(hdr,cnt.fs);
mnt= getElectrodePositions(cnt.clab);
grd= sprintf('F3,Fp1,Fz,Fp2,F4\nT5,C3,Cz,C4,T6\nP3,O1,Pz,O2,P4');
mnt= mnt_setGrid(mnt, grd);

cnt.hdr = hdr.Header;


%anonymization

id = strfind(cnt.hdr,'Filename = ');
id2 = find(cnt.hdr(id:end)==10); id2 = id+id2(1)-1;
id = id+length('Filename =');
cnt.hdr(id:id2-1) = ' ';


id = strfind(cnt.hdr,'PatientName = ');
id2 = find(cnt.hdr(id:end)==10); id2 = id+id2(1)-1;
id = id+length('PatientName =');
cnt.hdr(id:id2-1) = ' ';

id = strfind(cnt.hdr,'PatientId = ');
id2 = find(cnt.hdr(id:end)==10); id2 = id+id2(1)-1;
id = id+length('PatientId =');
cnt.hdr(id:id2-1) = ' ';

id = strfind(cnt.hdr,'PatientDob = ');
id2 = find(cnt.hdr(id:end)==10); id2 = id+id2(1)-1;
id = id+length('PatientDob =');
str = cnt.hdr(id:id2-1);
cnt.hdr(id:id2-1) = ' ';

id3 = strfind(str,'/');
year = str(id3(2)+1:id3(2)+4);
cnt.hdr(id+1:id+4) = year;


