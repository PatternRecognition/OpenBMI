function cmap= cmap_bluewhite(m, varargin)
%CMAP_BLUEWHITE - Colormap going from blue to white
%
%Usage:
%  MAP= cmap_bluewhite(M)
%
%Input:
%  M  : Size of the colormap (number of entries). Default value: Same size
%       as current colormap
%
%Output:
%  MAP: A colormap matrix of size [M 3]
%
%Example:
%  clf; 
%  colormap(cmap_bluewhite(15));
%  imagesc(toeplitz(1:15)); colorbar;
%
%See also COLORMAP, HSV2RGB, CMAP_HSV_FADE

if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'min_sat', 0.25, ...
                  'min_val', 0);

m1= floor(m/2);
m2= m-m1;
map1= cmap_hsv_fade(m1+1, 4/6, 1, [opt.min_sat 1]);
map2= cmap_hsv_fade(m2+1, 4/6, [1 opt.min_val], 1);

cmap= [map1; map2(3:end,:)];
