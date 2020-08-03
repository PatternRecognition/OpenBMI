function eegfile_writeBVheader(file, varargin)
% EEGFILE_WRITEBVHEADER - Write Header in BrainVision Format
%
% Synopsis:
%   eegfile_writeBVheader(FILE, 'Property1', Value1, ...)
%
% Arguments:
%   FILE: string containing filename to save in.
%  
% Properties: 
%   'fs': sampling interval of raw data, required
%   'clab': cell array, channel labels, required
%   'scale': scaling factors for each channel, required
%   'DataPoints': number of datapoints, required
%   'precision': precision (default 'INT16')
%   'data_file': name of corresponding .eeg file (default FILE)
%   'marker_file': name of corresponding .mrk file (default FILE)
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
                  'precision', 'INT16', ...
                  'data_file', fileName, ...
                  'marker_file', fileName);

if ~ischar(opt.DataPoints),  %% not sure, why DataPoints is a string
  opt.DataPoints= sprintf('%d', opt.DataPoints);
end

fid= fopen([fullName '.vhdr'], 'w','b');
if fid==-1, error(sprintf('cannot write to %s.vhdr', fullName)); end
fprintf(fid, ['Brain Vision Data Exchange Header File Version 1.0' 13 10]);
fprintf(fid, ['; Data exported from BBCI Matlab Toolbox' 13 10]);
fprintf(fid, [13 10 '[Common Infos]' 13 10]);
fprintf(fid, ['DataFile=%s.eeg' 13 10], opt.data_file);
fprintf(fid, ['MarkerFile=%s.vmrk' 13 10], opt.marker_file);
fprintf(fid, ['DataFormat=BINARY' 13 10]);
fprintf(fid, ['DataOrientation=MULTIPLEXED' 13 10]);
fprintf(fid, ['NumberOfChannels=%d' 13 10], length(opt.clab));
fprintf(fid, ['DataPoints=%s' 13 10], opt.DataPoints);
fprintf(fid, ['SamplingInterval=%g' 13 10], 1000000/opt.fs);
fprintf(fid, [13 10 '[Binary Infos]' 13 10]);
switch(lower(opt.precision)),
 case 'int16',
  fprintf(fid, ['BinaryFormat=INT_16' 13 10]);
  fprintf(fid, ['UseBigEndianOrder=NO' 13 10]);
 case 'int32',
  fprintf(fid, ['BinaryFormat=INT_32' 13 10]);
  fprintf(fid, ['UseBigEndianOrder=NO' 13 10]);
 case {'float32','single','float'},
  fprintf(fid, ['BinaryFormat=IEEE_FLOAT_32' 13 10]);
 case {'float64','double'},
  fprintf(fid, ['BinaryFormat=IEEE_FLOAT_64' 13 10]);
 otherwise,
  error(['Unknown precision, not implemented yet: ' opt.precision]);
end
fprintf(fid, [13 10 '[Channel Infos]' 13 10]);
for ic= 1:length(opt.clab),
  fprintf(fid, ['Ch%d=%s,,%g' 13 10], ic, opt.clab{ic}, ...
          opt.scale(min(ic,end)));
end
fprintf(fid, ['' 13 10]);

if isfield(opt, 'impedances')
   fprintf(fid, ['Impedance [kOhm].' 13 10]);
   for ic= 1:length(opt.clab)
      if isinf(opt.impedances(ic))
          fprintf(fid, [opt.clab{ic} ':   Out of Range!' 13 10]);
      else
          fprintf(fid, [opt.clab{ic} ':   ' num2str(opt.impedances(ic)) 13 10]);
      end
   end
end

fclose(fid);
