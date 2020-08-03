function map= cmap_rainbow(m, varargin)
%CMAP_RAINBOW - Colormap going from red to violet
%
%Description:
%  This function returns a colormap. The colors range from red over
%   yellow, green and blue to violet.
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
%  clf; imagesc(toeplitz(1:50)); colorbar;
%  colormap(cmap_rainbow);
% 
%See also COLORMAP, HSV2RGB, CMAP_HSV_FADE

%blanker@first.fhg.de, 01/2005

if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'sat', 1, ...
                  'val', 1);

map= cmap_hsv_fade(m, [0 21/24], [1 1]*opt.sat, [1 1]*opt.val);
