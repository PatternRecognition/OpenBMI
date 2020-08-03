function [cnt,mrk]= readKEHEEG(file, chans, fs, from, maxlen)
%[cnt,mrk]=readKEHEEG(file, <chans, fs, from, maxlen>)
%[cnt,mrk]=readKEHEEG(file, <chans, fs, ival>)
%
% IN   file  - file name (no extension),
%              relative to EEG_RAW_DIR unless beginning with '/' (resp '\')
%      chans - channels to load (labels or indices), [] means all
%      fs    - sampling interval, must be an integer divisor of the
%              sampling interval of the raw data, default: 128
%              fs may also be 'raw' which means sampling rate of raw signals
%      ival  - interval to read, [start end] in msec
%      from  - start [ms] for reading
%      maxlen- maximum length [ms] to read
%      tec_ch- give back technical channels or not. default: 0
%
% OUT  cnt      struct for contiuous signals
%         .x    - EEG signals (time x channels)
%         .clab - channel labels
%         .fs   - sampling interval
%      mrk     marker structure
%
% Note: Although there are usually two technical channels in the data, 
%       as can be seen in the clab and the PRS.Kanalanzahl_gespeichert - 
%       number from the header, they are not passed on to cnt.
%
% GLOBZ  EEG_RAW_DIR

%% Benjamin Blankertz, FhG-FIRST/kraulem 28/10/04

global EEG_RAW_DIR

cellSize= 4;  %% signed int32 -> 4 bytes

if ~exist('fs', 'var') | isempty(fs), fs='raw'; end
if ~exist('from', 'var') | isempty(from), from=0; end
if ~exist('maxlen', 'var') | isempty(maxlen),
  if length(from)==2,
    maxlen= diff(from);
    from= from(1);
  else
    maxlen=inf; 
  end
end
if ~exist('tec_ch','var')
  tec_ch = 0;
end

if file(1)==filesep,
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

[cnt.clab, scale, raw_fs, endian,len,cnt.PRS,cnt.ERS, cnt.ELS]= readKEHHeader(file);
if isequal(fs, 'raw'),
  fs= raw_fs;
end
lag= raw_fs/fs;
if lag~=round(lag) | lag<1,
  error('fs must be a positive integer divisor of the file''s fs');
end
cnt.fs= fs;
nChans = cnt.PRS.Kanalzahl_gespeichert;

% clab might be permuted:
d = [cnt.ELS.Reihenfolge];
if any(find(~d))
  clab = cnt.clab;
  [clab{d(find(d))}] = deal(cnt.clab{find(d)});
  e = setdiff(1:nChans,d);
  [clab{e}] = deal(cnt.clab{find(~d)});
  cnt.clab = clab;
end

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
skip= floor(from/1000*raw_fs);
nSamples= floor((fileLen - (cnt.PRS.Seekposition_Datenbeginn-1))/cellSize/nChans) - skip;
nSamples= min(nSamples, maxlen/1000*raw_fs);
T= floor(nSamples/lag);
uChans= length(chInd);

offset= cellSize * (skip + (ceil(lag/2)-1)) * nChans + (cnt.PRS.Seekposition_Datenbeginn);

if uChans==nChans,
  fseek(fid, offset, 'bof');
  prec= sprintf('%d*bit32', nChans);
  cnt.x= fread(fid, [nChans T], prec, cellSize * (lag-1) * nChans * 8);
  % Note: *8 is necessary since fread usually asks for bytes in the skip-argument,
  % unless 'precision' contains 'bit'.
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
                 fread(fid, [1 T], 'bit32', cellSize * (lag*nChans-1) * 8);
  end
end
fclose(fid);
cnt.x= cnt.x';
cnt.title= file;
if ~tec_ch
  cnt = proc_selectChannels(cnt,'not','K1','K2');
end
% now just add the marker:
if nargout>1
  [mrk,classDef] = readKEHMarker(file,fs);
  mrk = makeClassMarkers(mrk,classDef,0,0);
end