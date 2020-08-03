function bbci= bbci_log_setHeaderInfo(bbci, msg)
%BBCI_LOG_SETHEADERINFO - Set line in the header to be written in the log file
%
%Synopsis:
%  BBCI= bbci_log_setHeaderInfo(BBCI, MSG)


if ~isfield(bbci, 'log');
  bbci.log= struct;
end
bbci.log.header= {msg};
