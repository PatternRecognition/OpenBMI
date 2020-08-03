function [ht, ax]= showERtvalueGrid(epo, mnt, alpha, yLim, varargin)
%[ht, ax]= showERtvalueGrid(eeg, mnt, <alpha, yLim, FLAGS>)
%
% IN   epo     - struct of epoched signals, see makeSegments
%      mnt     - struct for electrode montage, see setElectrodeMontage
%      alpha   - significance level, default 0.01
%      yLim    - common y limits of all axes, or
%                  '-': y limits should be chosen individually, or
%                   []: global y limit chosen automatically, default
%      FLAGS   - 
%                'squared': square before averaging
%
% OUT  ht      - handle of title string
%      ax      - handle of subaxes
%

% bb, GMD-FIRST 09/00

bbci_obsolete(mfilename, 'grid_plot');

if ~exist('yLim', 'var') | isempty(yLim), yLim=[]; end
if ~exist('alpha', 'var') | isempty(alpha), alpha= 0.01; end

colorOrder= get(gca, 'colorOrder');
clf;
flags= {'small', varargin{:}};

dispChans= find(~isnan(mnt.box(1,1:end-1)) & ...
                ismember(strhead(mnt.clab), strhead(epo.clab)));
ax= zeros(1, length(dispChans));
for ic= dispChans,
  ax(ic)= axes('position', getAxisGridPos(mnt, ic));
  set(ax(ic), 'colorOrder',colorOrder);
  hold on;
  nEvents= showERtvalues(epo, mnt, mnt.clab{ic}, flags{:});
  if ic==dispChans(1),
    nu= sum(nEvents)-2;      %% degrees of freedom
    t_crit= calcTcrit(alpha, nu);
    xLim= get(gca, 'xLim');
  end
  line(xLim, t_crit*[1 1], 'color',0.2*[1 1 1]);
  line(xLim, -t_crit*[1 1], 'color',0.2*[1 1 1]);
  box on;
  hold off;
end
ht= showScalpLabels(epo, mnt, nEvents, yLim, ax);
