function cnt= readGenericEEG_float(file, chans, fs, from, maxlen)
%cnt=readGenericEEG_float(file, <chans, fs, from, maxlen>)
%
% IN   file  - file name (no extension),
%              relative to EEG_RAW_DIR unless beginning with '/' (resp '\')
%      chans - channels to load (labels or indices), [] means all
%      fs    - sampling interval, must be an integer divisor of the
%              sampling interval of the raw data, default: 100
%              fs may also be 'raw' which means sampling rate of raw signals
%      from  - start [ms] for reading
%      maxlen- maximum length [ms] to read
%
% OUT  cnt      struct for contiuous signals
%         .x    - EEG signals (time x channels)
%         .clab - channel labels
%         .fs   - sampling interval
%
% input arguments {from, to} are not implemented yet
%
% GLOBZ  EEG_RAW_DIR

%% Benjamin Blankertz, FhG-FIRST

global EEG_RAW_DIR

cellSize= 4;  %% 'float' -> 4 bytes

if ~exist('fs', 'var') | isempty(fs), fs=100; end
if ~exist('from', 'var') | isempty(from), from=0; end
if ~exist('maxlen', 'var') | isempty(maxlen), maxlen=inf; end

if file(1)==filesep,
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

[cnt.clab, scale, raw_fs, endian]= readGenericHeader(file, 'float');
if isequal(fs, 'raw'),
  fs= raw_fs;
end
lag= raw_fs/fs;
if lag~=round(lag) | lag<1,
  error('fs must be a positive integer divisor of the file''s fs');
end
cnt.fs= fs;
nChans= length(cnt.clab);
if ~exist('chans','var') | isempty(chans), 
  chInd= 1:nChans;
else
  chInd= unique(chanind(cnt.clab, chans));
  cnt.clab= {cnt.clab{chInd}};
end

fid= fopen([fullName '.eeg'], 'r', endian);
if fid==-1, error(sprintf('%s.eeg not found', fullName)); end

fseek(fid, 0, 'eof');
fileLen= ftell(fid);
skip= from/1000*raw_fs;
nSamples= floor(fileLen/cellSize/nChans) - skip;
nSamples= min(nSamples, maxlen/1000*raw_fs);
T= floor(nSamples/lag);
uChans= length(chInd);

offset= cellSize * (skip + (ceil(lag/2)-1)) * nChans;
if uChans==nChans,
  fseek(fid, offset, 'bof');
  prec= sprintf('%d*float', nChans);
  cnt.x= fread(fid, [nChans T], prec, cellSize * (lag-1)*nChans);
  for ci= 1:uChans,
    cn= chInd(ci);    %% this loopy version is faster than matrix multiplic.
    cnt.x(ci,:)= scale(cn)*cnt.x(ci,:);
  end
else
  cnt.x= zeros(uChans, T);
  for ci= 1:uChans,
    cn= chInd(ci);
    fseek(fid, offset+(cn-1)*cellSize, 'bof');
    cnt.x(ci,:)= scale(cn) * ...
                 fread(fid, [1 T], 'float', cellSize * (lag*nChans-1));
  end
end
fclose(fid);

cnt.x= cnt.x';
cnt.title= file;
