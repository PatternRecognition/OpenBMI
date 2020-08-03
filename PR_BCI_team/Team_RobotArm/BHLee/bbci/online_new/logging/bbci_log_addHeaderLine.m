function bbci= bbci_log_addHeaderLine(bbci, msg)
%BBCI_LOG_ADDHEADERLINE - Add line to the header to be written in the log file
%
%Synopsis:
%  BBCI= bbci_log_addHeaderLine(BBCI, MSG)


if ~isfield(bbci, 'log');
  bbci.log= struct;
end
if ~isfield(bbci.log, 'header'),
  bbci.log.header= {};
end

dim= min(find(size(bbci.log.header)>1));
if isempty(dim), 
  dim= 1;
end
bbci.log.header= cat(dim, bbci.log.header, msg);
