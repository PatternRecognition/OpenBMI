function [bbci, data]= bbci_calibrate(bbci, data)
%BBCI_CALIBRATE - Calibrate BBCI classifier and features to given data
%
%Synopsis:
%  [BBCI, DATA]= bbci_calibrate(BBCI, DATA)
%
%To get a description on the structures 'BBCI' and 'DATA', type
%help bbci_calibrate_structures

% 09-2011 Benjamin Blankertz


bbci= bbci_calibrate_setDefaults(bbci);
BC= bbci.calibrate;

if exist('data','var') && isfield(data,'cnt'),
  data.isnew= 0;  
else
  data = bbci_load(bbci);
end

data.log= bbci_log_open(BC.log);

% Log info about calibration files and BBCI settings
bbci_log_write(data, '#Calibration files from folder <%s>:', BC.folder);
file_counter = 1;
for k= 1:length(data.fileinfo),
    for f = 1:length(data.fileinfo{k}),
      msg= sprintf('File %d: %s <%s>, size %d', file_counter, data.fileinfo{k}(f).name, ...
                   data.fileinfo{k}(f).date, data.fileinfo{k}(f).bytes);
      bbci_log_write(data, ['#' msg]);
      file_counter = file_counter + 1;
    end
end
bbci_log_write(data, '\n#Settings of BBCI:');
bbci_prettyPrint(data.log.fid, copy_fields(bbci, 'calibrate'));

% Store original values for recovery via bbci_calibrate_reset
if ~isfield(bbci, 'default_settings'),
  bbci.calibrate.default_settings= bbci.calibrate.settings;
end

[bbci, data]= BC.fcn(bbci, data);
data.previous_settings= bbci.calibrate.settings;

bbci_log_write(data, '\n#Calibrated BBCI online system:');
bbci_prettyPrint(data.log.fid, rmfield(bbci, 'calibrate'));
bbci_log_close(data);
