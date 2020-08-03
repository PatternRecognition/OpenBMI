function mrk= readMarkerTable(mrkName, fs, markerTypes, flag)
%mrk= readMarkerTable(mrkName, <fs=100, markerTypes, flag>)
%
% IN   mrkName    - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 100. Use 'raw' for the original sampling rate.
%      markerTypes- read on markers of this type,
%                   default {'Stimulus','Response'}
%      flag       - default [1 -1], i.e. response markers give positive
%                   marker numbers and stimulus markers give negative
%
% OUT  mrk        struct for event markers
%         .toe    - type of event
%         .pos    - position in data points (for lagged data)
%         .fs     - sampling interval
%
% This works only for Stimuli and response marker,
% See readMarkerComment for comment markers
%
% GLOBZ  EEG_RAW_DIR

global EEG_RAW_DIR

if ~exist('fs', 'var'), fs=100; end
if ~exist('markerTypes', 'var'), markerTypes={'Stimulus','Response'}; end
if ~exist('flag', 'var'), flag=[1 -1]; end
markerTypes= lower(markerTypes);

if (isunix & mrkName(1)==filesep) | (~isunix & mrkName(2)==':')
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end
if isequal(fs, 'raw'),
%  [dmy, dmy, fs]= readGenericHeader(fullName);
  hdr= eegfile_readBVheader(fullName);
  fs= hdr.fs;
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
mrk.toe= [];
ei= 0;
while ~feof(fid),
  str= fgets(fid);
  if isempty(str) | str(1)==';', continue; end
  [mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    strread(str, 'Mk%u=%s%s%u%u%u%s', 'delimiter',',');
  mi= strmatch(lower(mrktype), markerTypes, 'exact');
  if ~isempty(mi),
    ei= ei+1;
    mrk.pos(ei)= ceil(pos/lag);
    nums= desc{1}(2:end);
%    if (desc{1}(1)=='R' | desc{1}(1)=='S') & ischar(nums),
    if ischar(nums),
      mrk.toe(ei)= flag(mi)*str2num(nums);
    else
      warning('unrecognized description format');
      mrk.toe(ei)= 1;
    end
  end
end
mrk.fs= fs;

fclose(fid);



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
