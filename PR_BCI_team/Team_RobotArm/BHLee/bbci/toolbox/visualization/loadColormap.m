function map = loadColormap(mapName)
%STAT load a common colormap from colormaps.mat without a figure pop up
%
% IN: mapName = string of a colormap name 
%
% like 'hsv','hot','gray','bone','copper','pink','white','flag','lines',
% 'colorcube','vga','jet','prism','cool','autumn','spring','winter','summer
% '
% SEE ALSO: saveColormap

if ~exist('colormaps.mat','file')
    error('First generate colormaps.mat with saveColormap !');
end
load colormaps.mat
colornames = {'hsv','hot','gray','bone','copper','pink','white','flag','lines','colorcube','vga','jet','prism','cool','autumn','spring','winter','summer'};
mapName=lower(mapName);
for i=1:size(colormaps,2),
	if strcmp(mapName,colornames{i}),
		map=colormaps{i};
		break;
	end
end
