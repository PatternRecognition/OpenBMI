function mrk = readMarkerTiming(file,fs)
% Reads out file to get timing of the marker file + segments
%
% GUido Dornhege, 12/10/05

if ~exist('fs','var') || isempty(fs);
  fs = 100;
end

global EEG_RAW_DIR

if isunix 
  if file(1)~='/'
    file = [EEG_RAW_DIR file];
  end
else
  if file(2)~=':'
    file = [EEG_RAW_DIR file];
  end
end

if isequal(fs, 'raw'),
  [dmy, dmy, fs]= readGenericHeader(file);
end

fid= fopen([file '.vhdr'], 'r');
if fid==-1, error(sprintf('%s.vhdr not found', file)); end
mrk_fs=  1000000/str2num(getEntry(fid, 'SamplingInterval='));
datp = str2num(getEntry(fid,'DataPoints='));

lag= mrk_fs/fs;
fclose(fid);

mrk = struct('length',floor(datp/mrk_fs*fs));
fid = fopen([file '.vmrk'],'r');
getEntry(fid, '[Marker Infos]');

infos = {};
while ~feof(fid);
  str = fgets(fid);
  if isempty(str) || str(1)==';', continue; end
  [mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    strread(str, 'Mk%u=%s%s%u%u%u%s', 'delimiter',',');
  
  if strcmp(mrktype,'New Segment');
    infos = cat(1,infos,{mrkno,pos,seg_time{:}});
  end
end
fclose(fid);

mrk.pos = [infos{:,2}]';
mrk.time = zeros(length(mrk.pos),6);

for i = 1:length(mrk.pos)
  str = infos{i,3};
  mrk.time(i,1) = str2num(str(1:4));
  mrk.time(i,2) = str2num(str(5:6));
  mrk.time(i,3) = str2num(str(7:8));
  mrk.time(i,4) = str2num(str(9:10));
  mrk.time(i,5) = str2num(str(11:12));
  mrk.time(i,6) = str2num(str(13:end))/(10^(length(str)-14));
end

mrk.pos = ceil(mrk.pos/lag);
mrk.fs = fs;


function entry= getEntry(fid, keyword)

fseek(fid, 0, 'bof');
ok= 0;
while ~ok && ~feof(fid),
  str= fgets(fid);
  ok= strncmp(keyword, str, length(keyword));
end
if keyword(end)=='=',
  entry= str(length(keyword)+1:end);
end
