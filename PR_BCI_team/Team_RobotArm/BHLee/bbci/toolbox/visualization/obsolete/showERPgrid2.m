function [ht, ax, hleg, he]= showERPgrid2(epo, mnt, yLim, varargin)
%[ht, ax]= showERPhead2(epo, mnt, <yLim, FLAGS>)
%
% IN   epo     - struct of epoched signals, see makeSegments
%      mnt     - struct for electrode montage, see setElectrodeMontage
%      yLim    - common y limits of all axes, or
%                  '-': y limits should be chosen individually, or
%                   []: global y limit chosen automatically, default
%      FLAGS   - see showERP
%
% OUT  ht      - handle of title string
%      ax      - handle of subaxes
%
% SEE  showERP, makeSegments, setElectrodeMontage

% bb, FhG-FIRST 04/02

bbci_obsolete(mfilename, 'grid_plot');

if ~exist('yLim', 'var'), yLim=[]; end

colorOrder= get(gca, 'colorOrder');
clf;
set(gcf, 'color',[1 1 1]);
flags= {'legend', 'small', varargin{:}};

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end

axesStyle= {'color',0.5*[1 1 1], 'linewidth',0.2};
ax= zeros(1, length(dispChans));
hl= zeros(2, length(dispChans));
for ic= dispChans,
%  ax(ic)= axes;
  ax(ic)= axes('position', getAxisGridPos(mnt, ic));
  set(ax(ic), 'colorOrder',colorOrder);
  if isfield(epo, 'refIval'),
    hp(ic)= patch(epo.refIval([1 2 2 1]), [0 0 0 0], 0.8*[1 1 1]);
  end
  hl(:,ic)= line([0 0; 0 0], [0 0; 0 0], axesStyle{:});
  hold on;
  [nEvents, dummy, hc]= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  if ic==dispChans(1),
%    set(hc, 'position', getAxisGridPos(mnt, 0));
    flags= {flags{2:end}}; 
    hleg= hc;
  end
%  set(ax(ic), 'position', getAxisHeadPos(mnt, ic, axisSize));
  axis off;
  hold off;
end


if ~exist('yLim', 'var') | isempty(yLim) | isequal(yLim,'sym'),
  nonEEG= chanind(mnt, 'E*');
  displayedEEG= setdiff(dispChans, nonEEG);
  if isempty(displayedEEG),
    displayedEEG= dispChans;
  end
  yLim= unifyYLim(ax(displayedEEG));
  if isequal(yLim,'sym'),
    yl= max(abs(yLim));
    yLim= [-yl yl];
  end
end

axisShrink= 0.5;
yAx= yLim + [1 -1]*axisShrink*diff(yLim)/2;
leg_pos= getAxisGridPos(mnt, 0);
if ~any(isnan(leg_pos)),
  axes('position', leg_pos);
  xAx= epo.t([1 end]);
  line([[xAx(1);0],[0;0]], [[0;0],[0;yAx(2)]], 'color','k');
  %hh= text(xAx(1)/2, 0.03*yAx(2), sprintf('%d ms', round(-xAx(1))));
  %set(hh, 'horizontalAli','center', 'verticalAli','bottom');
  hh= text(0.1*xAx(1), 0.03*yAx(2), sprintf('%d ms', round(-xAx(1))));
  set(hh, 'horizontalAli','right', 'verticalAli','bottom');
  hh= text(0.1*xAx(2), yAx(2)/2, sprintf('%d \\muV', round(yAx(2))));
  set(hh, 'verticalAli','middle');
  hh= text(0.1*xAx(2), yAx(2), '+');
  set(hh, 'verticalAli','middle');
  hh= text(0.1*xAx(2), 0, '-');
  set(hh, 'verticalAli','middle');
  set(gca, 'xLim',xAx, 'yLim',yLim);
  axis off;

  legpos= get(hleg, 'position');
  legpos(2)= legpos(2) - 0.5*legpos(4);
  set(hleg, 'position',legpos);
  legend('ResizeLegend');
end

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
%    set(hp(ic), 'yData',yPatch([1 1 2 2]), 'edgeColor',0.8*[1 1 1]);
    set(hp(ic), 'yData',yPatch([1 1 2 2]), 'edgeColor','none');
  end
end
set(he, 'verticalAlignment','bottom');
%set(he, 'verticalAlignment', 'top');

if isfield(epo, 'title'),
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
  tit= [untex(epo.title) ':  ' tit];
  ht= addTitle(tit, 1);
end
