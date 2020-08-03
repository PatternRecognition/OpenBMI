function mrk= readSegmentBorders(mrkName, fs)
%mrk= readSegmentBorders(mrkName, <fs=100>)
%
% This function reads the segment borders from the generic data format
% file. Use getSegmentBorders to load them from data in bbci matlab format.
%
% IN   mrkName    - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 100
% OUT  mrk        struct for event markers
%         .ival   - each row is a two dimensional vector specifying
%                   a segment [begin end] (in data points)
%         .fs     - sampling interval
%
% See getSegmentBorders
%
% GLOBZ  EEG_RAW_DIR


global EEG_RAW_DIR

if ~exist('fs', 'var') || isempty(fs), fs=100; end

if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end

fid= fopen([fullName '.vhdr'], 'r');
if fid==-1, error('%s.vhdr not found', fullName); end
mrk_fs=  1000000/str2num(getEntry(fid, 'SamplingInterval='));
if isequal(fs, 'raw'),
  fs= mrk_fs;
  lag= 1;
else
  lag= mrk_fs/fs;
end
fclose(fid);


fid= fopen([fullName '.vmrk'], 'r');
if fid==-1, error('%s.vmrk not found', fullName); end

getEntry(fid, '[Marker Infos]');

mrk_pos= [];
ei= 0;
while ~feof(fid),
  str= fgets(fid);
  if isempty(str) || str(1)==';', continue; end
  [mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    strread(str, 'Mk%u=%s%s%u%u%u%s', 'delimiter',',');
  if strcmp(mrktype, 'New Segment')
    ei= ei+1;
    mrk_pos(ei)= ceil(pos/lag);
  end
end
mrk.fs= fs;

fclose(fid);

if isempty(mrk_pos),
  warning('segment infos found');
  return;
end

[d,d,d,d, len]= readGenericHeader(fullName);
nSeg= length(mrk_pos);
iv= [1:nSeg-1; 2:nSeg]';
mrk.ival= [mrk_pos(iv); mrk_pos(end) ceil(len*fs)+1];
mrk.ival(:,2)= mrk.ival(:,2)-1;


function entry= getEntry(fid, keyword)

if keyword(1)=='[', fseek(fid, 0, 'bof'); end
ok= 0;
while ~ok && ~feof(fid),
  str= fgets(fid);
  ok= strncmp(keyword, str, length(keyword));
end
if keyword(end)=='=',
  entry= str(length(keyword)+1:end);
end

