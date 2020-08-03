function map= green_black_red(m, v, hue1, hue2)
% green_black_red - Colormap going from green over black to red
%
% Synopsis:
%   map = green_black_red(m,v,hue1,hue2)
%   
% Arguments:
%   m: Size of the colormap (number of entries). Default value: Same size as
%       current colormap
%   v: Scalar. V value for the colormap in the HSV color model. Default: 1
%   hue1: Scalar or [1 3] vector. H value for the 'green' entries in the HSV
%       color model. If of size [1 3], this is taken to be the RGB triple for the
%       starting 'green' values. Default: 0.
%   hue2: Scalar or [1 3] vector. H value for the 'red' entries in the HSV color
%       model. If of size [1 3], this is taken to be the RGB triple for the starting
%       'red' values. Default: 0
%   
% Returns:
%   map: A colormap matrix of size [M 3]
%   
% Description:
%   This colormap goes from green over black to red. This is often used
%   to display microarray image data.
%   The colormap is produced by linear interpolation between the given
%   colors hue1 and hue1 in the RGB color space. This is in contrast to
%   green_white_red, which interpolates in the HSV color space.
%   
% Examples:
%   colormap(green_black_red)
%     does this in the expected way.
%   green_black_red(128, 0.5)
%     produces a 128-entry map that starts from dark green, going to dark red
%   green_black_red([],1,0.1,0.8)
%     gives a colormap that goes from orange over black to purple.
%   green_black_red([],1,[1 1 0], [0 0 1])
%     gives a colormap that goes from yellow over black to blue.
%   
% See also: green_white_red,colormap,hsv2rgb
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: green_black_red.m,v 1.1 2005/03/02 12:07:07 neuro_toolbox Exp $

error(nargchk(0, 4, nargin));

if nargin<4 | isempty(hue2),
  hue2 = 0;
end
if nargin<3 | isempty(hue1),
  hue1 = 1/3;
end
if nargin<2 | isempty(v),
  v = 1;
end
if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1); 
end

if prod(size(hue1))==1,
  % Scalar argument: assume this is a hue value in HSV model. Convert to
  % RGB triple
  rgb1 = hsv2rgb([hue1 1 v]);
elseif length(hue1)==3,
  rgb1 = hue1(:)';
else
  error('Input argument HUE1 must be scalar or an RGB triple');
end
if prod(size(hue2))==1,
  % Scalar argument: assume this is a hue value in HSV model. Convert to
  % RGB triple
  rgb2 = hsv2rgb([hue2 1 v]);
elseif length(hue2)==3,
  rgb2 = hue2(:)';
else
  error('Input argument HUE2 must be scalar or an RGB triple');
end

m1 = ceil(m/2);
m2 = m-m1;
if m1<=0,
  map= [];
else
  map = zeros([m 3]);
  % Interpolate linearly from color 1 to black
  map(1:m1,:) = (1-linspace(0, 1, m1)')*rgb1;
  % Interpolate linearly from black to color 2
  map(m1+1:end, :) = linspace(0, 1, m2)'*rgb2;
end
