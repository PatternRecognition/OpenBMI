function data_log= bbci_log_open(bbci_log, fid);
%BBCI_LOG_OPEN - Open log file, if logging to file is requested
%
%Synopsis:
%  DATA_LOG= bbci_log_open(BBCI_LOG, <FID>);
%
%Arguments:
%  BBCI_LOG - Field 'log' of the BBCI structure (see bbci_apply_structures)
%             which specifies the details about logging.
%
%Returns:
%  DATA_LOG - Structure holding information about the log file to be
%             stored in the DATA structure (see bbci_apply_structures)

% 02-2011 Benjamin Blankertz


if nargin<2,
  data_log= [];
end

switch(bbci_log.output),
 case {0, 'none'},
  data_log.fid= [];
 case 'screen',
  data_log.fid= 1;
 case {'file', 'screen&file'},
  if nargin>1 && ~isempty(fid) && max(fid)>2,
    % log file is already open
    if strcmp(bbci_log.output, 'file'),
      data_log.fid= max(fid);
    else
      data_log.fid= [1 max(fid)];
    end
    return;
  end
  if isabsolutepath(bbci_log.file),
    log_filename= bbci_log.file;
  else
    log_filename= fullfile(bbci_log.folder, bbci_log.file);
  end
  data_log.filename= [log_filename '.txt'];
  if ~bbci_log.force_overwriting,
    num= 1;
    while exist(data_log.filename, 'file'),
      num= num + 1;
      data_log.filename= sprintf('%s%03d.txt', log_filename, num);
    end
  end
  data_log.fid= fopen(data_log.filename, 'w');
  if data_log.fid==-1,
    error(sprintf('could not open log-file <%s> for writing', log_filename));
  end
  if strcmp(bbci_log.output, 'screen&file'),
    data_log.fid= [1, data_log.fid];
  end
  if isfield(bbci_log, 'header'),
    timestr=  datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');
    bbci_log.header{1}= strrep(bbci_log.header{1}, '<TIME>', timestr);
    bbci_log_write(data_log.fid, '%s', bbci_log.header{:});
  end
 otherwise,
  error('log.output not recognized');
end
