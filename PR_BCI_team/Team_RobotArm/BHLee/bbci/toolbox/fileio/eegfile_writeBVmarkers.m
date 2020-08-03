function eegfile_writeBVmarkers(file, mrk, varargin)
% EEGFILE_WRITEBVMARKERS - Write Markers in BrainVision Format
%
% Synopsis:
%   eegfile_writeBVmarkers(FILENAME, MRK, 'Property1',Value1, ...)
%
% Arguments:
%   FILE: string containing filename to save in.
%   MRK: marker structure.
%  
% Properties:
%     'data_file': name of the corresponding .eeg file (default FILENAME)
%
% See also: eegfile_*
%

global EEG_EXPORT_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'export_dir', EEG_EXPORT_DIR);

if (isunix & file(1)==filesep) | (~isunix & file(2)==':')
  fullName= file;
else
  fullName= [opt.export_dir '/' file];
end

[pathstr, fileName]= fileparts(fullName);
opt= set_defaults(opt, ...
                  'data_file', fileName);

fid= fopen([fullName '.vmrk'], 'w');
if fid==-1, error(sprintf('cannot write to %s.vmrk', fullName)); end

fprintf(fid, ['Brain Vision Data Exchange Marker File, Version 1.0' 13 10]);
fprintf(fid, [13 10 '[Common Infos]' 13 10]);
fprintf(fid, ['DataFile=%s.eeg' 13 10], opt.data_file);
fprintf(fid, [13 10 '[Marker Infos]' 13 10]);
if exist('mrk', 'var') & ~isempty(mrk),
  if ~isfield(mrk,'type'),
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
  elseif length(mrk)==1
    for im = 1:length(mrk.type)
      fprintf(fid, ['Mk%d=%s,%s,%u,%u,%u,%s' 13 10], im, ...
              mrk.type{im}, mrk.desc{im}, mrk.pos(im), ...
              mrk.length(im), mrk.chan(im), mrk.time{im});
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
