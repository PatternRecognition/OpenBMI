function mrk= readAlternativeMarkers(mrkName, classDef, fs)
%mrk= readAlternativeMarker(file_name, classDef, <fs>)
%
% IN   file_name  - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      classDef   - e.g. {{'Response','R  1'}, {'Scanner','Scan Start'};
%                         'key', 'scan start'};         
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 100
% OUT  mrk        struct for event markers
%         .str    - cell array containing the comments
%         .pos    - position in data points (for lagged data)
%         .fs     - sampling interval
%
%      if input argument classDef is given .str is left out, but
%         .y      - class label
%
% GLOBZ  EEG_RAW_DIR

global EEG_RAW_DIR

if ~exist('fs', 'var') | isempty(fs), fs=100; end

if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end

fid= fopen([fullName '.vhdr'], 'r');
if fid==-1, error(sprintf('%s.vhdr not found', fullName)); end
mrk_fs=  1000000/str2num(getEntry(fid, 'SamplingInterval='));
lag= mrk_fs/fs;
fclose(fid);


fid= fopen([fullName '.vmrk'], 'r'); 
if fid==-1, error(sprintf('%s.vmrk not found', fullName)); end

getEntry(fid, '[Marker Infos]');

mrkType = cat(1,classDef{1,:});
entries = mrkType(:,2);
mrkType = mrkType(:,1);

mrk.pos= [];
ei= 0;
while ~feof(fid),
  str= fgets(fid);
  if isempty(str) | str(1)==';', continue; end
  [mno,mtype,desc,pos,pnts,chan,seg_time]= ...
    strread(str, 'Mk%u=%s%s%u%u%u%s', 'delimiter',',');
    ind = find(strcmp(mtype, mrkType));
    if ~isempty(ind)
      class_no= ind(strmatch(desc, entries(ind)));
      if ~isempty(class_no),
        ei= ei+1;
        mrk.pos(ei)= ceil(pos/lag);
        mrk.toe(ei)= class_no;
      end
    end
end
fclose(fid);

if isempty(mrk.pos),
  warning('no markers found');
  return;
end

nClasses= size(classDef,2);
mrk.fs= fs;
mrk.y= zeros(nClasses, length(mrk.pos));
for ic= 1:nClasses,
  mrk.y(ic,:)= (mrk.toe==ic);
end

if size(classDef,1)>1,
  mrk.className= {classDef{2,:}};
end



function entry= getEntry(fid, keyword)

if keyword(1)=='[', fseek(fid, 0, 'bof'); end
ok= 0;
while ~ok & ~feof(fid),
  str= fgets(fid);
  ok= strncmp(keyword, str, length(keyword));
end
if keyword(end)=='=',
  entry= str(length(keyword)+1:end);
end
