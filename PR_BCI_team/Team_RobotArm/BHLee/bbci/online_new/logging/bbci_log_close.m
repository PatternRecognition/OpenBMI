function bbci_log_close(data)
%BBCI_LOG_CLOSE - Close log file(s)
%
%Synopsis:
%  bbci_log_close(DATA)
%
%Arguments:
% DATA - DATA strcut as in bbci_apply, bbci_calibrate. The subfield
%        DATA.log.fid holds the file identifiers of the log files.

% 12-2011 Benjamin Blankertz


if isfield(data, 'log'),
  fid_to_close= data.log.fid(find(data.log.fid>2));
  for fid= fid_to_close,
    fclose(fid);
  end
end
