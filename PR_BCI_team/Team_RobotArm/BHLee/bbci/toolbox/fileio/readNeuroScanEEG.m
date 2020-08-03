function cnt= readNeuroScanEEG(file, chans, fs, from, maxlen)
%cnt=readNeuroScanEEG(file, <chans, fs, from, maxlen>)
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
% GLOBZ  EEG_RAW_DIR
%
% NOT TESTED SO FAR !!!

%% Benjamin Blankertz, FhG-FIRST

global EEG_RAW_DIR

cellSize= 2;  %% 'short' -> 2 bytes

if ~exist('fs', 'var') | isempty(fs), fs=100; end
if ~exist('from', 'var') | isempty(from), from=0; end
if ~exist('maxlen', 'var') | isempty(maxlen), maxlen=inf; end

if file(1)==filesep,
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

[cnt.clab, raw_fs, len, cal, sen, bas, bsz]= readNeuroScanHeader(file);
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

fid= fopen([fullName '.cnt'], 'r', 'ieee-le');
if fid==-1, error(sprintf('%s.cnt not found', fullName)); end

header_length= 900 + 75*nChans;
fseek(fid, 0, 'eof');
fileLen= ftell(fid) - header_length;
skip= from/1000*fs;
nSamples= floor(fileLen/cellSize/nChans) - skip;
nSamples= min(nSamples, maxlen/1000*fs);
T= floor(nSamples/lag);
uChans= length(chInd);

offset= header_length + cellSize * (skip + (ceil(lag/2)-1)) * nChans;
if uChans==nChans,
  fseek(fid, offset, 'bof');
  prec= sprintf('%d*short', nChans);
  cnt.x= fread(fid, [nChans T], prec, cellSize * (lag-1)*nChans);
else
  cnt.x= zeros(uChans, T);
  for ci= 1:uChans,
    cn= chInd(ci);
    fseek(fid, offset+(cn-1)*cellSize, 'bof');
    qq= fread(fid, [1 T], 'short', cellSize * (lag*nChans-1));
    cnt.x(ci,:)= qq;
  end
end
fclose(fid);

scale= cal(chInd).*sen(chInd)/204.8;
for ch= 1:uChans,
  cnt.x(ch,:)= (cnt.x(ch,:) - bas(chInd(ch))) * scale(ch);
end

cnt.x= cnt.x';
cnt.title= file;
