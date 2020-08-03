function [h, h_background]= stimutil_initMsg_sssep(varargin)
%STIMUTIL_INITMSG - Generate a Text Object for Displaying Messages
%
%Synopsis:
% HANDLE= stimutil_initMsg(<OPT>)
%
%Arguments:
% OPT: struct or property/value list of optional arguments:
% 'handle_msg': Handle to text object which is used to display the countdown
%    message. If empty a new object is generated. Default [].
% 'handle_background': Handle to axis object on which the message should be
%    rendered. If empty a new object is generated. Default [].
% 'msg_vpos': Scalar. Vertical position of message text object. Default: 0.57.
% 'msg_spec': Cell array. Text object specifications for message text object.
%   Default: {'FontSize',0.1, 'FontWeight','bold', 'Color',[.3 .3 .3]})
%
%Returns:
% HANDLE: Handle to text object.

% blanker@cs.tu-berlin.de, Jul-2007
% modified by Fouad, april-2008

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'handle_msg', [], ...
                  'handle_background', [], ...
                  'msg_vpos', 0.05, ...
                  'msg_spec', {'FontSize',0.1, 'FontWeight','bold', ...
                               'Color',0.3*[1 1 1]});

if isempty(opt.handle_background),
  opt.handle_background= axes('Position',[0 0 1 1]);
  set(opt.handle_background, 'XLim',[-1 1], 'YLim',[-1 1], 'Visible','off');
end
axes(opt.handle_background);
if isempty(opt.handle_msg);
  opt.handle_msg= text(0, opt.msg_vpos, strvcat('Read this text while','the stimulation is on.','And keep reading it...'));
end
msg_spec= {'HorizontalAli','center', ...
           'VerticalAli','middle', ...
           'FontUnits','normalized', ...
           'Visible','on', ...
           opt.msg_spec{:}};
set(opt.handle_msg, msg_spec{:});

if nargout>0,
  h= opt.handle_msg;
end
if nargout>1,
  h_background= opt.handle_background;
end
