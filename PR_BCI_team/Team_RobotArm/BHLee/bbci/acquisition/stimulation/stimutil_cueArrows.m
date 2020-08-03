function [H, H_cross]= stimutil_cueArrows(dirs, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'cross', 0);

for dd= 1:length(dirs),
  if dd<length(dirs),
    h_tmp= stimutil_drawArrow(dirs{dd}, opt, 'cross',0);
  else
    h_tmp= stimutil_drawArrow(dirs{dd}, opt);
  end
  H(dd)= h_tmp.arrow;
end

if opt.cross,
  H_cross= h_tmp.cross;
else
  H_cross= [];
end

set([H H_cross], 'Visible','off');
