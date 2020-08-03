function [H,opt]= stimutil_fixationCross(varargin)
%STIMUTIL_FIXATIONCROSS - Initialize Fixation Cross for Cue Presentation
%
%H= stimutil_fixationCross(<OPT>)
%
%Arguemnts:
% OPT: struct or property/value list of optional properties:
%
%Returns:
% H - Handle to graphic objects

% blanker@cs.tu-berlin.de, Nov 2007


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'handle_background', [], ...
                  'cross_size', 0.15, ...
                  'cross_width', 0.02, ...
                  'cross_color', 0*[1 1 1], ...
                  'cross_edgecolor', 'none');
%                  'cross_vpos', 0.12, ...

if isempty(opt.handle_background),
  opt.handle_background= stimutil_initFigure(opt);
end


%fix_w= opt.cross_size;
%fix_h= fix_w;

%H= line([-fix_w fix_w; 0 0]', ...
%        opt.cross_vpos + [0 0; -fix_h fix_h]', ...
%        opt.cross_spec{:}, ...
%        'Visible','off');

c= opt.cross_size;
v= opt.cross_width;
H= patch([v v c c v v -v -v -c -c -v -v], ...
               [c v v -v -v -c -c -v -v v v c], opt.cross_color);
set(H, 'EdgeColor', opt.cross_edgecolor);
