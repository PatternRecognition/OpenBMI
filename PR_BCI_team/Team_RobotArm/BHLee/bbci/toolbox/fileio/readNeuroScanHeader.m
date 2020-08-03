function [Clab, fs, len, cal, sen, bas, bsz]= readNeuroScanHeader(fileName)
%[Clab, fs, len, cal, sen, bas, bsz]= readNeuroScanHeader(fileName)
%
% IN:  file - name of cnt file (without extension)
%
% OUT: Clab - channel labels (cell array of strings)
%      fs   - sampling rate
%      len  - length of EEG channels in samples
%      cal  - calibration for each channel
%      sen  - sensitivity of each channel
%      bas  - baseline offset for each channel
%
% many entries are NOT CHECKED
%
% GLOBZ  EEG_RAW_DIR

if fileName(1)==filesep,
  fullName= fileName;
else
  global EEG_RAW_DIR
  fullName= [EEG_RAW_DIR fileName];
end

fid= fopen([fullName '.cnt'], 'r', 'ieee-le');
if fid==-1, error(sprintf('%s.cnt not found', fullName)); end

fseek(fid, 0, 'eof');
fLen= ftell(fid);
fseek(fid, 370, 'bof');
nChans= fread(fid, 1, 'int16');
fseek(fid, 376, 'bof');
fs= fread(fid, 1, 'uint16');
fseek(fid, 886, 'bof');
et= fread(fid, 1, 'uint32');
fseek(fid, 892, 'bof');
bsz= fread(fid, 1, 'uint32');
offset= 900;
len= (et - offset - 75*nChans) / (2*nChans);
Clab= cell(1, nChans);
for ch=1:nChans
  off= offset + 75*(ch-1);
  fseek(fid, off, 'bof');
  lab= fread(fid, 10, 'char');
  Clab{ch}= char(lab(1:max(find(lab>0)))');
end
fseek(fid, offset + 47, 'bof');
bas= fread(fid, nChans, 'int16', 73);
fseek(fid, offset + 59, 'bof');
sen= fread(fid, nChans, 'float32', 71);
fseek(fid, offset + 71, 'bof');
cal= fread(fid, nChans, 'float32', 71);

fclose(fid);
