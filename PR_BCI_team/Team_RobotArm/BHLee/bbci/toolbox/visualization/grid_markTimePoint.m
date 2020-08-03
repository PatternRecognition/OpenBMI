function hl = grid_markTimePoint(tp, chans, varargin)
%grid_markTimePoint(tp, <chans, linespec>)
%
% IN  tp     - time point [msec]
%     chans  - channels which should be marked, default [] meaning all

if ~exist('chans','var'), chans=[]; end

hsp= grid_getSubplots(chans);
for ih= hsp,
  axes(ih);
  yl= get(ih, 'yLim');
  hl= line([tp tp], yl, varargin{:});
  moveObjectBack(hl);
end
