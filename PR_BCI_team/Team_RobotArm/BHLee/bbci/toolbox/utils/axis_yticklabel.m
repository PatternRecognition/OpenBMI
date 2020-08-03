function ht= axis_yticklabel(label_list, varargin)
%AXIS_YTICKLABEL - Add y-ticklabels to the current axis
%
%Synopsis:
% H= axis_yticklabel(LABEL_LIST, <OPT>)
%
%Input:
% LABEL_LIST: string (labels separated by '|') or cell array of strings
% OPT: struct or property/value list of optional properties:
%  .hpos - horizontal position
%  .ytick - position of yticks. If not provided, get(gca,'YTick') is used.
%  .color - font color in RGB format. Maybe an [nLabels 3] matrix of color
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

% Author(s): Benjamin Blankertz

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'ytick', [], ...
                 'hpos', -0.03, ...
                 'color', [0 0 0], ...
                 'HorizontalAlignment','right', ...
                 'VerticalAlignment','middle');  
%% Matlab7 bug?? VerticalAli='middle' is not in the middle on the screen,
%% but on print.

if ~iscell(label_list),
  if ismember('|',label_list),
    label_str= label_list;
    idx= find(label_str=='|');
    idx= [0, idx, length(label_str)+1];
    nLab= length(idx)-1;
    label_list= cell(1,nLab);
    for ii= 1:nLab,
      label_list{ii}= label_str(idx(ii)+1:idx(ii+1)-1);
    end
  else
    label_list= {label_list};
  end
end
nLab= length(label_list);
if isdefault.ytick,
  opt.ytick= get(gca, 'YTick');
end
if length(opt.ytick)~=nLab,
  error('number of labels must match number of xticks');
end
if nLab>1 & size(opt.color,1)==1,
  opt.color= repmat(opt.color, [nLab 1]);
end

opt_fn= fieldnames(opt);
ifp= find(ismember(opt_fn, {'HorizontalAlignment','VerticalAlignment'}));
ifp= cat(1, ifp, strmatch('font', opt_fn));
font_opt= struct_copyFields(opt, opt_fn(ifp));
font_pl= struct2propertylist(font_opt);

XLim= get(gca, 'XLim');
xx= XLim(1) + opt.hpos*diff(XLim);
yy= opt.ytick;

for tt= 1:nLab,
  ht(tt)= text(xx, yy(tt), label_list{tt});
  set(ht(tt), font_pl{:}, 'Color',opt.color(tt,:));
end
