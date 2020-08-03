function saveColormap()
%STAT saves all common colormaps to colormaps.mat
%
% can be loaded with loadColormap('mapName') again without a figure pop up
%
% SEE ALSO: loadColormap
colornames = {'hsv','hot','gray','bone','copper','pink','white','flag','lines','colorcube','vga','jet','prism','cool','autumn','spring','winter','summer'};
for i=1:size(colornames,2), colormaps{i} = colormap(colornames{i});close all,end
save colormaps.mat colormaps
