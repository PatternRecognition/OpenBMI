function hl= grid_markRange(yRange, chans, varargin)
%hl= grid_markRange(yRange, <chans, linespec>)
%
% IN  yRange - range ([lower upper]) on y axis [uV]
%     chans  - channels which should be marked, default [] meaning all

if ~exist('chans','var'), chans=[]; end
if length(yRange)==1, yRange=[-yRange yRange]; end

hsp= grid_getSubplots(chans);
for ih= hsp,
  axes(ih);
  xl= get(ih, 'xLim');
  hl= line(xl, yRange([1 1]), varargin{:});
  moveObjectBack(hl);
  hl= line(xl, yRange([2 2]), varargin{:});
  moveObjectBack(hl);
end
