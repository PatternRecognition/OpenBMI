function bbci= acq_stoprecording(bbci, file, varargin)
%ACQ_STOPRECORDING - Stop data acquisition
%
%Synopsis:
%  BBCI= acq_stoprecording(BBCI, FILE)


pause(0.05);

fcn= strrep(func2str(bbci.source.acquire_fcn), 'bbci_acquire_', '');
switch(fcn),
 case 'bv',
  bvr_sendcommand('stoprecording');
% case 'sigserv',
%  ppTrigger(254);
 otherwise,
  error('this command is not implemented for this acquisition device');
end
