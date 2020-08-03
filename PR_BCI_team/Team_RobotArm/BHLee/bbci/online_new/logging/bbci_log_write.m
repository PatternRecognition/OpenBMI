function bbci_log_write(fid, format, varargin)
%BBCI_LOG_WRITE - Print information to screen or file
%
%Synopsis:
%  bbci_log_write(FID, FORMAT, PARAM1, ...)
%  bbci_log_write(DATA, FORMAT, PARAM1, ...)
%
%Arguments:
%  FID    - Vector of file identifier, e.g. [1 file_id] for printing log
%           log information to the screen and to the file of ID file_id.
%  DATA   - DATA struct as in bbci_calibate and bbci_apply. The FID of
%           the log file is in the subfield DATA.log.fid.
%  FORMAT - As for fprintf.
%  PARAMx - Parameters passed to fprintf.


if isstruct(fid),  % First input argument is given as DATA
  fid= fid.log.fid;
end

for k= 1:length(fid),
  fprintf(fid(k), [format '\n'], varargin{:});
end
