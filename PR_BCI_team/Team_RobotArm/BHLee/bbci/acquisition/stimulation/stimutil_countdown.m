function h= stimutil_countdown(duration, varargin)
%STIMUTIL_COUNTDOWN - Show Countdown
%
%Synopsis:
% stimutil_showCountdown(DURATION, <OPT>)
% 
%Arguments:
% DURATION: Duration of countdown in seconds.
% OPT: struct or property/value list of optional arguments:
% 'countdown_msg': Default: 'start in %s s'
% 'handle': Handle to text object which is used to display the countdown
%    message. If empty a new object is generated.
%
%Returns:
% HANDLE: Handle to text object.

% blanker@cs.tu-berlin.de, Jul-2007

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'handle_msg', [], ...
                  'countdown_msg', '%d', ...
                  'countdown_fontsize', []);
%                  'countdown_msg', 'Start in %d s');

opt.handle_msg= stimutil_initMsg(opt);
memo.fontsize= get(opt.handle_msg, 'FontSize');
if ~isempty(opt.countdown_fontsize),
  set(opt.handle_msg, 'FontSize', opt.countdown_fontsize);
end

waitForSync;
for ii= duration:-1:1,
  msg= sprintf(opt.countdown_msg, ii);
  set(opt.handle_msg, 'String', msg);
  drawnow;
  waitForSync(1000);
end
set(opt.handle_msg, 'String','', 'FontSize',memo.fontsize);

if nargout>0,
  h= opt.handle_msg;
end
