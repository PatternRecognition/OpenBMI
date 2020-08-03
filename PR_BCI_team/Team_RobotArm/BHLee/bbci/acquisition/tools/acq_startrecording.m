function bbci= acq_startrecording(bbci, file, varargin)
%ACQ_STARTRECORDING - Start data acquisition
%
%Synopsis:
%  BBCI= acq_startrecording(BBCI, FILE)


fcn= strrep(func2str(bbci.source.acquire_fcn), 'bbci_acquire_', '');
switch(fcn),
 case 'bv',
  filename= bvr_startrecording(file, 'impedances',0, varargin{:});
 case 'sigserv',
  filename= signalServer_startrecoding(file, varargin{:});
 otherwise,
  error('this acquisition device is not handled');
end

bbci= bbci_log_setHeaderInfo(bbci, ['# EEG file: ' filename]);

pause(0.05);
