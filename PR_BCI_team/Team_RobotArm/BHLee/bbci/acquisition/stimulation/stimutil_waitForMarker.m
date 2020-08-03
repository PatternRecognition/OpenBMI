function marker = stimutil_waitForMarker(varargin)
%STIMUTIL_WAITFORMARKER - Wait until specified marker is received
%
%Synopsis:
% stimutil_waitForMarker(<OPT>)
%or
% stimutil_waitForMarker(STOPMARKERS)
%as shorthand for
% stimutil_waitForMarker('stopmarkers', STOPMARKERS)
% 
%Arguments:
% OPT: struct or property/value list of optional arguments:
% 'stopmarkers', string or cell array of strings which specify those
%     marker types that are awaited, e.g., {'R  1','R128'}. Using 'S' or
%     'R' as stopmarkers matches all Stimulus resp. Response markers.
%     OPT.stopmarker can also be a vector of integers, which are interpreted
%     as stimulus markers. Default: 'R*'.
% 'bv_host': IP or host name of computer on which BrainVision Recorder
%      is running, default 'localhost'.
% 'bv_bbciclose': true or false. If true, perform initially bbciclose.

% blanker@cs.tu-berlin.de, Jul-2007

global acquire_func; 

if mod(nargin,2)==1 & ~isstruct(varargin{1}),
  stopmarkers= varargin{1};
  opt= propertylist2struct(varargin{2:end});
  opt.stopmarkers= stopmarkers;
else
  opt= propertylist2struct(varargin{:});
end
[opt,isdefault]= set_defaults(opt, ...
                  'stopmarkers', 'R', ...
                  'bv_host', 'localhost', ...
                  'bv_bbciclose', 0, ...
                  'fs', 1000, ...
                  'state',[],...
                  'pause',0.05, ...
                  'verbose', 0);

if isequal(acquire_func, @acquire_sigserv),
  [opt,isdefault]= opt_overrideIfDefault(opt, isdefault, 'pause',0);
  if isdefault.fs,
    % get sampling rate from signal server
    [sig_info, dmy, dmy]= mexSSClient('localhost',9000,'tcp');
    opt.fs= sig_info(1);
    acquire_func('close');
  end
end

if isnumeric(opt.stopmarkers),
  opt.stopmarkers= cprintf('S%3d', opt.stopmarkers);
end

if isempty(opt.bv_host),
  fprintf('Waiting for marker disabled by OPT. Press any key to continue.\n');
  pause;
  return;
end

if opt.bv_bbciclose,
  bbciclose;
end

if opt.verbose,
  fprintf('connecting to acquisition system\n');
end

if isdefault.state
  opt.state= acquire_func(opt.fs, opt.bv_host);
  opt.state.reconnect= 1;
  [dmy]= acquire_func(opt.state);  %% clear the queue
end

if opt.verbose,
  fprintf('waiting for marker %s\n', toString(opt.stopmarkers));
end

stopmarker= 0;
while ~stopmarker,
  if opt.verbose>2,
    fprintf('%s: acquiring data\n', datestr(now,'HH:MM:SS.FFF'));
  end
  [dmy,dmy,dmy,mt,dmy]= acquire_func(opt.state);
  if ~isempty(mt) && opt.verbose>1,
    fprintf('%s: received markers: %s\n', datestr(now,'HH:MM:SS.FFF'), vec2str(mt));
  end
  for mm= 1:length(mt),
    if ~isempty(strmatch(mt{mm}, opt.stopmarkers)),
      stopmarker= 1;
      if opt.verbose,
        fprintf('stop marker received: %s\n', mt{mm});
      end
    end
  end
  pause(opt.pause);  %% this is to allow breaks
end
acquire_func('close');

if nargin>0
  marker = mt{mm};
end
