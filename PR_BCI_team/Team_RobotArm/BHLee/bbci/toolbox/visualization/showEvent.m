function [hp,hl]= showEvent(epo, ev, mrk, scaleFactor)
%hp= showEvent(epo, ev, <mrk, scaleFactor=1>)

if ~exist('scaleFactor','var'), scaleFactor=1; end

nChans= size(epo.x,2);
col= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);

cla;
hold on;
set(gca, 'colorOrder',col);

T= size(epo.x,1);
baseline= mean(epo.x(:,:,ev));
Base= ones(T,1) * baseline;
Shift= ones(T,1) * (1:nChans);
scale= 1./std(epo.x(:,:,ev));
%scale(find(isinf(scale)))= 1;
scale(find(scale>1))= 1;       %% exclude 'constant' channels from scaling calc
scale(:)= median(scale)/10 * scaleFactor;
eeg_traces= (Base - epo.x(:,:,ev))*diag(scale) + Shift;
hp= plot(epo.t, eeg_traces);
xLim= epo.t([1 end]);
set(gca, 'xLim',xLim, ...
    'yLim',[0 nChans+1], 'yTick',1:nChans, 'yTickLabel',epo.clab);
hl= line(xLim'*ones(1,nChans), [1;1]*(1:nChans));
set(hl, 'color','k', 'lineStyle',':');
hold off;
axis ij;

if exist('mrk', 'var') & ~isempty(mrk),
  [so,si]= sort(mrk.pos);
  mrk_so= pickEvents(mrk, si);
  it= find(si==ev);
  iShow= it;
  ie= it;
  while ie>1 & (mrk_so.pos(ie-1)-mrk_so.pos(it))*1000/mrk.fs>epo.t(1),
    ie= ie-1;
    iShow= [iShow ie];
  end
  ie= it;
  while ie<length(mrk.pos) & ...
        (mrk_so.pos(ie+1)-mrk_so.pos(it))*1000/mrk.fs<epo.t(end),
    ie= ie+1;
    iShow= [iShow ie];
  end
  showMarker(mrk_so, iShow);
end

if nargout==0,
  clear hp;
end
