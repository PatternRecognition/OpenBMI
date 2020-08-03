function mrk= readNeuroScanMarkers(mrkName, fs, ext)
%mrk= readNeuroScanMarkers(mrkName, <fs=100, extension='evt'>)
%
% IN   mrkName    - name of marker file (without extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default 100.
%      extension  - of the event file, default 'evt'.
%
% OUT  mrk        struct for event markers
%         .toe    - type of event
%         .pos    - position in data points (for lagged data)
%         .fs     - sampling interval
%
% GLOBZ  EEG_RAW_DIR

global EEG_RAW_DIR


if ~exist('fs', 'var'), fs=100; end
if ~exist('ext', 'var'), ext='evt'; end

if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end

[dummy, raw_fs]= readNeuroScanHeader(mrkName);
if isequal(fs, 'raw'),
  fs= raw_fs;
end

try,
  [d, toe, d, d, d, pos]= ...
      textread([fullName '.' ext], '%d%d%d%d%f%d');
catch,
  error(sprintf('%s not found', fullName));
end

mrk.toe= toe';
mrk.pos= ceil(pos/raw_fs*fs)';
mrk.fs= fs;
