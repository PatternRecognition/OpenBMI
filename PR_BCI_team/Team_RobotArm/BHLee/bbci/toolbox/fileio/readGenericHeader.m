function [clab, scale, fs, endian, len]= ...
    readGenericHeader(hdrName, float_format)
%[clab, scale, fs, endian, len]= readGenericHeader(hdrName, float_format)
%
% IN   hdrName - name of header file (no extension),
%                relative to EEG_RAW_DIR unless beginning with '/'
%
% OUT  clab    - channel labels (cell array)
%      scale   - scaling factors for each channel
%      fs      - sampling interval of raw data
%      endian  - byte ordering: 'l' little or 'b' big
%      len     - length of the data set in seconds
%
% many entries are NOT CHECKED, e.g. DataFormat, DataOriantation, DataType
%
% GLOBZ  EEG_RAW_DIR

if ~exist('float_format', 'var'), float_format=[]; end

if (isunix & hdrName(1)==filesep) | (~isunix & hdrName(2)==':')
  fullName= hdrName;
else
  global EEG_RAW_DIR
  fullName= [EEG_RAW_DIR hdrName];
end

fid= fopen([fullName '.vhdr'], 'r'); 
if fid==-1, error(sprintf('%s.vhdr not found', fullName)); end


getEntry(fid, '[Common Infos]'); 
nChans= str2num(getEntry(fid, 'NumberOfChannels='));
fs= 1000000/str2num(getEntry(fid, 'SamplingInterval='));
if nargout>4,
  getEntry(fid, '[Common Infos]');
  dp= getEntry(fid, 'DataPoints=', 0);
  len= str2num(dp)/fs;
end

getEntry(fid, '[Binary Infos]');
if isempty(float_format) & ~isequal(getEntry(fid, 'BinaryFormat='), 'INT_16'),
  error('unsupported format');
end
isbig= getEntry(fid, 'UseBigEndianOrder=', 0);
if isequal(isbig, 'YES'),
  endian='b';
else
  endian='l';
end

  
getEntry(fid, '[Channel Infos]');
clab= cell(1, nChans);
scale= zeros(1, nChans);
ci= 0;
while ci<nChans,
  str= fgets(fid);
  if isempty(str) | str(1)==';', continue; end
  [chno,chname,refname,resol]= ...
    strread(str, 'Ch%u=%s%s%f', 'delimiter',',');
  ci= ci+1;
  clab{ci}= chname{1};
  scale(ci)= resol;
end

fclose(fid);



function entry= getEntry(fid, keyword, mandatory)

if ~exist('mandatory','var'), mandatory=1; end

if keyword(1)=='[', fseek(fid, 0, 'bof'); end
ok= 0;
while ~ok & ~feof(fid),
  str= fgets(fid);
  ok= strncmp(keyword, str, length(keyword));
end
if ~ok,
  if mandatory,
    error(sprintf('keyword <%s> not found', keyword));
  else
    entry= [];
    return;
  end
end
if keyword(end)=='=',
  entry= deblank(str(length(keyword)+1:end));
end
