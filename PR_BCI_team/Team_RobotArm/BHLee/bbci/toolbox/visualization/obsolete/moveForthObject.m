function moveForthObject(ho)
%moveForthObject(<ho>)
%
% IN  ho - handle of graphic object, default: last plotted object

bbci_obsolete(mfilename, 'moveObjectForth');

hc= get(gca, 'children');
if ~exist('ho', 'var') | isempty(ho),
  hi= length(hc);
else
  hi= find(hc==ho);
  if isempty(hi),
    error('handle not found in current axes');
  end
end

set(gca, 'children', [hc([hi, 1:hi-1, hi+1:end])]);
