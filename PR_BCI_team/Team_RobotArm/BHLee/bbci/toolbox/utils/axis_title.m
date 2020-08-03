function ht= axis_title(title_list, varargin)
%AXIS_TITLE - Add a title to the current axis
%
%Synopsis:
% H= axis_title(TITLE, <OPT>)
%
%Input:
% TITLE: string or cell array of strings
% OPT: struct or property/value list of optional properties:
%  .vpos - vertical position
%  .color - font color in RGB format. Maybe an [nTit 3] matrix of color
%           codes
%  .font* - font properties like fontWeight, fontSize, ...
%  .horizontalAlignment, .verticalAlignment
%
%Output:
% H: handle to the text object(s)
%
%Note:
% The position of the text object is defined *within* the axis. So you should
% set XLimMode and YLimMode to 'manual' before calling AXIS_TITLE.

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'vpos', 1.02, ...
                 'color', [0 0 0], ...
                 'horizontalAlignment','center', ...
                 'verticalAlignment','bottom');

if opt.vpos<=0 & isdefault.verticalAlignment,
  opt.verticalAlignment= 'top';
end

if ~iscell(title_list),
  title_list= {title_list};
end
nTit= length(title_list);
if nTit>1 & size(opt.color,1)==1,
  opt.color= repmat(opt.color, [nTit 1]);
end

opt_fn= fieldnames(opt);
ifp= find(ismember(opt_fn, {'horizontalAlignment','verticalAlignment'}));
ifp= cat(1, ifp, strmatch('font', opt_fn));
font_opt= struct_copyFields(opt, opt_fn(ifp));
font_pl= struct2propertylist(font_opt);

gap= 1/nTit;
xx= (gap/2):gap:1;
XLim= get(gca, 'XLim');
xx= XLim(1) + xx*diff(XLim);
YLim= get(gca, 'YLim');
yy= YLim(1) + opt.vpos*diff(YLim);

ht= text(xx, yy*ones(1,nTit), title_list);
set(ht, font_pl{:});
for tt= 1:nTit,
  set(ht(tt), 'color',opt.color(tt,:));
end
