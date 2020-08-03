function [ht, ax, hleg]= showERPhead(epo, mnt, yLim, varargin)
%[ht, ax]= showERPhead(epo, mnt, <yLim, FLAGS>)
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
% OUT  ht      - handle of title string
%      ax      - handle of subaxes
%
% SEE  makeSegments, setElectrodeMontage

% bb, FhG-FIRST 04/02

bbci_obsolete(mfilename, 'mnt_scalpToGrid + grid_plot');

if ~exist('yLim', 'var'), yLim=[]; end
axisSize= [0.17 0.14];

colorOrder= get(gca, 'colorOrder');
clf;
flags= {'legend', 'small', varargin{:}};

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
dispChans= intersect(dispChans, find(~isnan(mnt.x)));

ax= zeros(1, length(dispChans));
hl= zeros(2, length(dispChans));
for ic= dispChans,
%  ax(ic)= axes;
  ax(ic)= axes('position', getAxisHeadPos(mnt, ic, axisSize));
  set(ax(ic), 'colorOrder',colorOrder);
  if isfield(epo, 'refIval'),
    hp(ic)= patch(epo.refIval([1 2 2 1]), [0 0 0 0], 0.85*[1 1 1]);
  end
  hl(:,ic)= line([0 0; 0 0], [0 0; 0 0], 'color',0.5*[1 1 1], 'linewidth',0.2);
  hold on;
  [nEvents, dummy, hc]= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  if ic==dispChans(1),
    set(hc, 'position', getAxisHeadPos(mnt, 0, axisSize));
    flags= {flags{2:end}}; 
    hleg= hc;
  end
%  set(ax(ic), 'position', getAxisHeadPos(mnt, ic, axisSize));
  axis off;
  hold off;
end


if ~exist('yLim', 'var') | isempty(yLim),
  nonEEG= chanind(mnt, 'E*');
  displayedEEG= setdiff(dispChans, nonEEG);
  if isempty(displayedEEG),
    displayedEEG= dispChans;
  end
  yLim= unifyYLim(ax(displayedEEG));
end

axisShrink= 0.25;
yAx= yLim + [1 -1]*axisShrink*diff(yLim);
he= [];
for ic= dispChans,
  axes(ax(ic));
  xLim= get(gca, 'xLim');
  set(gca, 'yLim', yLim);
  x= xLim(1)+0.15*diff(xLim);
  ch= chanind(epo, mnt.clab(ic));
  he= [he text(x, yAx(2), epo.clab(ch))];
  set(hl(1,ic), 'xData',xLim', 'yData',[0;0]);
  set(hl(2,ic), 'xData',[0;0], 'yData',yAx');
  if isfield(epo, 'refIval'),
    yPatch= [-0.05 0.05] * diff(yLim);
    set(hp(ic), 'yData',yPatch([1 1 2 2]), 'edgeColor',0.85*[1 1 1]);
  end
end
set(he, 'verticalAlignment','bottom', 'fontSize',8);
%set(he, 'verticalAlignment', 'top');


if isfield(epo, 'className'),
  evtStr= [vec2str(epo.className, [], ' / ') ','];
else
  evtStr= '';
end
if isfield(epo, 't'),
  xLimStr= sprintf('[%g %g] ms ', trunc(epo.t([1 end])));
else
  xLimStr= '';
end
yLimStr= sprintf('[%g %g] \\muV', trunc(yLim));
tit= sprintf('%s N=%s,  %s %s', evtStr, vec2str(nEvents,[],'/'), ...
             xLimStr, yLimStr);
if isfield(epo, 'title'),
  tit= [untex(epo.title) ':  ' tit];
end
ht= addTitle(tit, 1);
