function mrk= readMarkerComments(mrkName, fs, classDef)
%mrk= readMarkerComments(mrkName, <fs=100, classDef>)
%
% IN   mrkName    - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 100
%      classDef   - makes classes from given comments, eg.
%                   {'Augen auf','Augen zu'};
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

mrk.pos= [];
mrk.str= {};
ei= 0;
while ~feof(fid),
  str= fgets(fid);
  if isempty(str) | str(1)==';', continue; end
  [mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    strread(str, 'Mk%u=%s%s%u%u%u%s', 'delimiter',',');
  if strcmp(mrktype, 'Comment')
    ei= ei+1;
    mrk.pos(ei)= ceil(pos/lag);
    mrk.str(ei)= desc;
  end
end
mrk.fs= fs;

fclose(fid);

if isempty(mrk.pos),
  warning('no markers found');
  return;
end

if exist('classDef','var'),
  nClasses= size(classDef,2);
  classInd= {};
  lab= [];
  for ic= 1:nClasses,
    classInd(ic)= {strmatch(classDef{1,ic}, mrk.str, 'exact')'};
    lab= [lab, ic*ones(size(classInd{ic}))];
  end
  [so, si]= sort([classInd{:}]);
  mrk.pos= mrk.pos(so);
  mrk.y= logical(zeros(nClasses, length(so)));
  for ic= 1:nClasses,
    mrk.y(ic,:)= lab(si)==ic;
  end
  mrk.className= {classDef{size(classDef,1),:}};
  mrk= rmfield(mrk, 'str');
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
