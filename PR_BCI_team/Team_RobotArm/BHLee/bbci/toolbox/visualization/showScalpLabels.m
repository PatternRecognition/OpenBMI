function [ht,he]= showScalpLabels(epo, mnt, nEvents, yLim, ax)
%[h_tit, h_clab]= showScalpLabels(epo, mnt, nEvents, yLim, ax)

% bb, GMD-FIRST 09/00

dispChans= find(ax~=0);
if ~exist('yLim', 'var') | isempty(yLim) | isequal(yLim,'sym'),
  nonEEG= chanind(mnt, 'E*','RES','Feedb','wheel','throttle','A1','A2');
  displayedEEG= setdiff(dispChans, nonEEG);
  if isempty(displayedEEG),
    displayedEEG= dispChans;
  end
  yL= unifyYLim(ax(displayedEEG));
  if isequal(yLim,'sym'),
    yl= max(abs(yL));
    yL= [-yl yl];
  end
  yLim= yL;
elseif isequal(yLim,'-')
  yLim= [];
end

he= [];
for ch= dispChans,
  axes(ax(ch));
  xLim= get(gca, 'xLim');
  if ~isempty(yLim), 
    set(gca, 'yLim', yLim);
    yl= yLim;
  else
    axis tight;
    yl= get(gca, 'yLim');
    yl(2)= yl(2)+0.1*diff(yl);
    yl(1)= truncsig(yl(1),2,'floor');
    yl(2)= truncsig(yl(2),2,'ceil');
    set(gca, 'yLim',yl);
    ht= text(xLim(2)-0.05*diff(xLim), yl(2), ...
             sprintf('[%.4g %.4g]', yl));
    set(ht, 'verticalAlignment','top', 'horizontalAlignment','right');
  end
  x= xLim(1)+0.02*diff(xLim);
%  he= [he text(x, yl(1), mnt.clab(ch))];
  ic= chanind(epo, mnt.clab(ch));
  he= [he text(x, yl(2), epo.clab(ic))];
  if isfield(epo, 'refIval'),
    yPatch= yl(1) + [0 0.05*diff(yl)];
    hp= patch(epo.refIval([1 2 2 1]), yPatch([1 1 2 2]), 0.85*ones(1,3));
    set(hp, 'lineWidth',0.1);
  end
end
%set(he, 'verticalAlignment', 'bottom');
set(he, 'verticalAlignment', 'top');

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
if isfield(epo, 'yUnit'),
  unit= epo.yUnit;
else
  unit= '\muV';
end
if isempty(yLim),
  yLimStr= 'individual yScale (!)';
else
  yLimStr= sprintf('[%g %g] %s', trunc(yLim), unit);
end
tit= sprintf('%s N=%s,  %s %s', evtStr, vec2str(nEvents,[],'/'), ...
             xLimStr, yLimStr);
if isfield(epo, 'title'),
  tit= [untex(epo.title) ':  ' tit];
end
ht= addTitle(tit);
