function map= green_white_red(m, v, hue1, hue2)
% green_white_red - Colormap going from green over white to red
%
% Description:
%   Returns a colormap
%
% Usage:
%   map = green_white_red
%   map = green_white_red(m, v, hue1, hue2)
%
% Input:
%   m: Size of the colormap (number of entries). Default value: Same size
%      as current colormap
%   v: V value for the colormap according to the HSV color
%      model. Default: 0.8 
%   hue1: H value for the 'green' entries, according to the HSV color
%      model. Default: 1/3
%   hue2: H value for the 'red' entries, according to the HSV color
%      model. Default: 0
%
% Output:
%   map: A colormap matrix of size [M 3]
%
% Examples:
%   map = green_white_red([], [], 0.1, 0.8)
%   returns a colormap going from orange over white to purple, using the
%   default values for m and v.
%
% See also colormap,hsv2rgb
%

% Original code from the BCI toolbox
% Documentation by Anton Schwaighofer, 08/2004
% $Id: green_white_red.m,v 1.2 2005/01/31 14:46:17 neuro_toolbox Exp $



%map= green_white_red(m)

if nargin<4 | isempty(hue2),
  hue2 = 0;
end
if nargin<3 | isempty(hue1),
  hue1 = 1/3;
end
if nargin<2 | isempty(v),
  v = 0.8;
end
if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1); 
end

mh= ceil(m/2);
if mh<=0,
  map= [];
else
  s= linspace(1, 0, mh)';
  o= ones(mh,1);
  map= [hsv2rgb([hue1*o s v*o]); hsv2rgb([hue2*o flipud(s) v*o])];
  if m<2*mh,
    map(mh,:)= [];
  end
end
