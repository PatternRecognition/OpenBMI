function stimutil_waitForMarker(varargin)
%STIMUTIL_WAITFORMARKER - Wait until specified marker is received
%
%Synopsis:
% stimutil_waitForMarker(<OPT>)
% 
%Arguments:
% OPT: struct or property/value list of optional arguments:
% 'stopmarkers', string or cell array of strings which specify those
%     marker types that are awaited, e.g., {'R  1','R128'}. String(s)
%     may contain wildcard '*' at the beginning or at the end (see
%     strinpatternmatch. Default: 'R*'.
% 'bv_host': IP or host name of computer on which BrainVision Recorder
%      is running, default 'localhost'.
% 'bv_bbciclose': true or false. If true, perform initially bbciclose.

% blanker@cs.tu-berlin.de, Jul-2007


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'stopmarkers', 'R*', ...
                  'bv_host', 'localhost', ...
                  'bv_bbciclose', 1);

if isempty(opt.bv_host),
  fprintf('Waiting for marker disabled by OPT. Press any key to continue.\n');
  pause;
  return;
end

if opt.bv_bbciclose,
  bbciclose;
end

state= acquire_bv(1000, opt.bv_host);
[dmy]= acquire_bv(state);  %% clear the queue

stopmarker= 0;
while ~stopmarker,
  [dmy,dmy,dmy,mt,dmy]= acquire_bv(state);
  for mm= 1:length(mt),
    if ~isempty(strpatternmatch(opt.stopmarkers, mt{mm})),
      stopmarker= 1;
    end
  end
  pause(0.001);  %% this is to allow breaks
end

bbciclose;
