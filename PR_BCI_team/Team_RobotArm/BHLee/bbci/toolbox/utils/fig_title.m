function H= fig_title(title, varargin)
%FIG_TITLE - Add a title to the current figure
%
%Synopsis:
% H= axis_fig(TITLE, <OPT>)
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


opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'vpos', 0.99, ...
                 'hpos', 0.5, ...
                 'color', [0 0 0], ...
                 'horizontalAlignment','center', ...
                 'verticalAlignment','top', ...
                 'fontWeight', 'bold', ...
                 'fontSize', 12);

if opt.vpos<=0 & isdefault.verticalAlignment,
  opt.verticalAlignment= 'bottom';
end

opt_fn= fieldnames(opt);
ifp= find(ismember(opt_fn, {'horizontalAlignment','verticalAlignment'}));
ifp= cat(1, ifp, strmatch('font', opt_fn));
font_opt= copy_struct(opt, opt_fn(ifp));
font_pl= struct2propertylist(font_opt);

xx= opt.hpos;
yy= opt.vpos;

H.axis= getBackgroundAxis;
H.text= text(xx, yy, title);
set(H.text, font_pl{:}, 'color',opt.color, 'Visible','on');
