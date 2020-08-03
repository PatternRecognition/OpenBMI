function map= cmap_linear(cols, n)
%CMAP_LINEAR - Colormap fading linearly between two colors
%
%Description:
%  This function returns a colormap. cols specify the colors included in
%  the interpolation. returns a colormap that linearly interpolates within
%  all colors (i.e. cols(i,:) )with n steps
%
%Usage:
%  MAP= cmap_linear(col_start, col_end, n)
%
%Input:
%  col_start: start color
%
%  col_end: end color
%
%  n: number of steps
%
%Output:
%  MAP: A colormap matrix of size [M 3]
%
%Example:
%  clf; imagesc(toeplitz(1:50))
%  colormap(cmap_linear([1 0 0;0 1 0], 128));
%  colorbar;
%
%See also COLORMAP, HSV2RGB
% Johannes 06/2011
n_col = size(cols,1);
n_sub= n / (n_col-1);

map = [];
linearTemplate = 0:1/(n_sub-1):1;
for ixcol = 1:n_col-1
    tmp_map = [];
    col_start = cols(ixcol,:);
    col_end = cols(ixcol+1,:);
    for ii = 1:3 %go through RGB     
        range = col_end(ii) - col_start(ii);
        tmp_map(:,ii) = linearTemplate * range + col_start(ii);
    end
    map = [map;tmp_map];
end