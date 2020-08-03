function cmap= cmap_greenwhitelila(m)
%CMAP_RAINBOW - Colormap going from green over white to lila
%
%Description:
%  This function returns a colormap. The colors goes from red to
%   yellow to white to cyan to blue.
%
%Usage:
%  MAP= cmap_rainbow(M)
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
%  colormap(cmap_posneg(65));
%  imagesc(toeplitz(1:65)); colorbar;
%
%See also COLORMAP, HSV2RGB, CMAP_HSV_FADE

%blanker@first.fhg.de, 01/2005

if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1);
end

if mod(m,2)==0,
  warning('m should be odd -> using m+1');
  m= m+1;
end

mb6= floor((m-1)/6);
mq= m-1 - 6*mb6;
m1= mb6;
m2= mb6 + (mq>=2);
m3= mb6 + (mq>=4);
m11= m1+ceil(m2/2);
m22= floor(m2/2);
map1= cmap_hsv_fade(m11+1, 2/6, 1, [0.5 1]);
map2= cmap_hsv_fade(m22+1, [2/6 1.3/6], 1, 1);
map3= cmap_hsv_fade(m3+1, 1.3/6, [1 0], 1);
map4= cmap_hsv_fade(m3+1, 5/6, [0 1], 1);
map5= cmap_hsv_fade(m2+1, [5/6 4.5/6], 1, 1);
map6= cmap_hsv_fade(m1+1, 4.5/6, 1, [1 0.5]);
cmap= [map1; map2(2:end,:); map3(2:end,:); ...
       map4(2:end,:); map5(2:end,:); map6(2:end,:)];
cmap= flipud(cmap);
