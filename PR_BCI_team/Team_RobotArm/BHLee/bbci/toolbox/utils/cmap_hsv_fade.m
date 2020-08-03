function map= cmap_hsv_fade(m, hue, sat, val)
%CMAP_HSV_FADE - Colormap fading between specified HSV values
%
%Description:
%  This function returns a colormap. For each parameter for the HSV
%  color model a start and an end value can be specified, between
%  which this function generates a linear interpolation.
%
%Usage:
%  MAP= cmap_hsv_fade(M, <HUE, SAT, VAL>)
%
%Input:
%  M  : Size of the colormap (number of entries). Default value: Same size
%       as current colormap
%  HUE: A pair of two values [HUE1 HUE2] according to the HSV color model.
%       The returned colormap linearly fades from HUE1 to HUE2.
%       If a scalar HUE is given, it is interpreted as [HUE HUE].
%  VAL: A pair of two values [VAL1 VAL2] according to the HSV color model.
%       The returned colormap linearly fades from VAL1 to VAL2.
%       If a scalar VAL is given, it is interpreted as [VAL VAL].
%  SAT: A pair of two values [SAT1 SAT2] according to the HSV color model.
%       The returned colormap linearly fades from SAT1 to SAT2.
%       If a scalar SAT is given, it is interpreted as [SAT SAT].
%
%Output:
%  MAP: A colormap matrix of size [M 3]
%
%Example:
%  clf; imagesc(toeplitz(1:50))
%  map1= cmap_hsv_fade(10, 1/6, [0 1], 1);
%  map2= cmap_hsv_fade(11, [1/6 0], 1, 1);
%  colormap([map1; map2(2:end,:)]);
%  colorbar;
% 
%See also COLORMAP, HSV2RGB

%blanker@first.fhg.de, 01/2005

if nargin<4 | isempty(val),
  val= 1;
end
if nargin<3 | isempty(sat),
  sat= [0 1];
end
if nargin<2 | isempty(hue),
  hue= 0;
end

if nargin<1 | isempty(m),
  m= size(get(gcf,'colormap'),1);
end
if length(hue)==1,
  hue= [hue hue];
end
if length(sat)==1,
  sat= [sat sat];
end
if length(val)==1,
  val= [val val];
end

map= hsv2rgb([mod(linspace(hue(1), hue(2), m), 1)' ...
              linspace(sat(1), sat(2), m)' ...
              linspace(val(1), val(2), m)']);
%% We use mod here for hue to account for the circular struture.
%% So it becomes possible to fade, e.g., from yellow (hue 1/6) to
%% purple (hue 5/6) going over red (hue 0) and not over green/blue
%% by using hue= [1/6 -1/6].
