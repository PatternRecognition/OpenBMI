function [h_tit, ax, h_clab]= showERPgrid(epo, mnt, yLim, varargin)
%[h_tit, ax, h_clab]= showERPgrid(epo, mnt, <yLim, FLAGS>)
%
% IN   epo     - struct of epoched signals, see makeSegments
%      mnt     - struct for electrode montage, see setElectrodeMontage
%      yLim    - common y limits of all axes, or
%                  '-': y limits should be chosen individually, or
%                   []: global y limit chosen automatically, default
%      FLAGS   - 
%        'diff':    plot differences between class averages
%        'squared': square before averaging
%
% OUT  h_tit   - handle of title string
%      h_ax    - handle of subaxes
%      h_clab  - handle of channel labels
%
% SEE  makeSegments, setElectrodeMontage

% bb, GMD-FIRST 09/00

bbci_obsolete(mfilename, 'grid_plot');

if ~exist('yLim', 'var'), yLim=[]; end

colorOrder= get(gca, 'colorOrder');
clf;
flags= {'legend', 'small', varargin{:}};

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end
ax= zeros(1, length(dispChans));
for ic= dispChans,
  ax(ic)= axes('position', getAxisGridPos(mnt, ic));
  set(ax(ic), 'colorOrder',colorOrder);
  hold on;
  nEvents= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  if ic==dispChans(1), flags={flags{2:end}}; end
  box on;
  hold off;
end
[h_tit, h_clab]= showScalpLabels(epo, mnt, nEvents, yLim, ax);
%set(gcf, 'resizeFcn','');
