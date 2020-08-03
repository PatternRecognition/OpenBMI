function cmap= cmap_whitered(m, varargin)

if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'min_sat', 0.4, ...
                  'min_val', 0, ...
                  'max_val', 0.8);

m1= floor(m/2.5);
m2= m-m1-1;
map1= cmap_hsv_fade(m1+1, 0, 1, [opt.min_sat 1]);
map2= cmap_hsv_fade(m2+1, 0, [opt.max_val opt.min_val], 1);

cmap= flipud([map1; map2(2:end,:)]);
