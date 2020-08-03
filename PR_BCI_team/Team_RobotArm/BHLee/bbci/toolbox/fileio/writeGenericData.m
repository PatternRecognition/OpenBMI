function writeGenericData(dat, mrk, scale)
%writeGenericData(dat, <mrk, scale>)
%
% IN  dat   - structure of continuous or epoch EEG data
%     mrk   - marker structure
%     scale - scaling factor used in the generic data format to bring
%             data from the int16 range (-32768 - 32767) to uV.
%             that means before saving signals are divided by
%             this factor.
%             individual scaling factors may be specified for each
%             channel in a vector, or a global scaling as scalar,
%             default is 0.1 (i.e. signal range is -3276.8 - 3276.7).
%
% use 
%   scale= max(abs(cnt.x))'/32768;
% to achive best resolution (least information loss in int16 conversion).
%
% C   readGenericEEG, readGenericHeader, readMarkerTable
%
% GLOBZ  EEG_EXPORT_DIR

file= dat.title;

if (isunix & file(1)==filesep) | (~isunix & file(2)==':')
  fullName= file;
else
  global EEG_EXPORT_DIR
  fullName= [EEG_EXPORT_DIR file];
end

[T, nChans, nEpochs]= size(dat.x);
if nEpochs>1,
  cnt= permute(dat.x, [2 1 3]);
  cnt= reshape(cnt, [nChans T*nEpochs]);
  if ~exist('mrk','var'),
    nEpochs= floor(size(dat.x,1)/T);
    mrk= [];
    mrk.pos= (1:nEpochs)*T;
    mrk.y= ones(1,nEpochs);
  end
else
  nEpochs= 0;
  cnt= dat.x';
end
if ~exist('scale', 'var'), scale=0.1; end
if length(scale)==1, scale= scale*ones(nChans,1); end
cnt= diag(1./scale)*cnt;
if any(cnt(:)>32767 | cnt(:)<-32768),
  warning('data clipped: use other scaling');
end

fid= fopen([fullName '.eeg'], 'wb');
if fid==-1, error(sprintf('cannot write to %s.eeg', fullName)); end
fwrite(fid, cnt, 'int16');
fclose(fid);

[pathstr, fileName]= fileparts(fullName);

fid= fopen([fullName '.vhdr'], 'w');
if fid==-1, error(sprintf('cannot write to %s.vhdr', fullName)); end
fprintf(fid, ['Brain Vision Data Exchange Header File Version 1.0' 13 10]);
fprintf(fid, ['; Data exported from matlab' 13 10]);
fprintf(fid, [13 10 '[Common Infos]' 13 10]);
fprintf(fid, ['DataFile=%s.eeg' 13 10], fileName);
fprintf(fid, ['MarkerFile=%s.vmrk' 13 10], fileName);
fprintf(fid, ['DataFormat=BINARY' 13 10]);
fprintf(fid, ['DataOrientation=MULTIPLEXED' 13 10]);
fprintf(fid, ['NumberOfChannels=%d' 13 10], nChans);
fprintf(fid, ['SamplingInterval=%g' 13 10], 1000000/dat.fs);
fprintf(fid, [13 10 '[Channel Infos]' 13 10]);
for ic= 1:nChans,
  fprintf(fid, ['Ch%d=%s,,%g' 13 10], ic, dat.clab{ic}, scale(min(ic,end)));
end
fprintf(fid, [13 10 '[Binary Infos]' 13 10]);
fprintf(fid, ['BinaryFormat=INT_16' 13 10]);
fprintf(fid, ['' 13 10]);
fclose(fid);

fid= fopen([fullName '.vmrk'], 'w');
if fid==-1, error(sprintf('cannot write to %s.vmrk', fullName)); end

fprintf(fid, ['Brain Vision Data Exchange Marker File, Version 1.0' 13 10]);
fprintf(fid, [13 10 '[Common Infos]' 13 10]);
fprintf(fid, ['DataFile=%s.eeg' 13 10], fileName);
fprintf(fid, [13 10 '[Marker Infos]' 13 10]);
if exist('mrk', 'var') & ~isempty(mrk),
  if length(mrk)==1,
    %% simple case, mrk is a usual (bbci) marker struct
    fprintf(fid, ['Mk1=New Segment,,1,1,0,00000000000000000000' 13 10]);
    for ie= 1:length(mrk.pos),
      if isfield(mrk, 'toe'),
        mt= mrk.toe(ie);
      else
        [d, mt]= max(mrk.y(:,ie));
      end
      if mrk.toe(ie)>0,
        fprintf(fid, ['Mk%d=Stimulus,S%3d,%d,1,0' 13 10], ie+1, mt, ...
                mrk.pos(ie));
      else
        fprintf(fid, ['Mk%d=Response,R%3d,%d,1,0' 13 10], ie+1, abs(mt), ...
                mrk.pos(ie));
      end
    end
  else
    for im= 1:length(mrk),
      fprintf(fid, ['Mk%d=%s,%s,%u,%u,%u,%s' 13 10], im, ...
              mrk(im).type, mrk(im).desc, mrk(im).pos, ...
              mrk(im).length, mrk(im).chan, mrk(im).time);
    end
  end
end
fclose(fid);
