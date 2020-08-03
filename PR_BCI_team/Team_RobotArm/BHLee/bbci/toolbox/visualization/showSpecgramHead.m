function [ax] = showSpecgramHead(dat, mnt, band, tLim, yLim)
% [ax] = showSpecgramHead(dat, mnt, band, tLim, yLim)
%
% visualizes the result of proc_specgram on the scalp at the
% electrode positions according to the eeg-montage
%
%   IN: dat    - special structure similar to epo or cnt (output of proc_specgram)
%       mnt    - eeg-montage containing the channel names and positions
%       band   - frequency range to display [Hz]
%       tLim   - time interval to display [ms]
%       yLim   - amplitude range for the color-coding
%
%  OUT: ax     - axis handle
%
% in case of open questions read the source code or contact
%
% stl, Berlin Aug. 2004
%
% see also proc_specgram

[nFreq, nTime, nChan] = size(dat.x) ; 
  
if ~exist('yLim', 'var'), yLim=[]; end
if ~exist('tLim', 'var') | isempty(tLim), 
  timeIdx = 1: nTime;
else
  timeIdx = find(dat.t>=tLim(1) & dat.t<=tLim(end));
end ;
if ~exist('band', 'var') | isempty(band),
  bandIdx = 1: nFreq ;
else
  bandIdx = find(dat.f>=band(1) & dat.f<=band(end)) ;
end


axisSize= [0.17 0.14];
clf;
dispChans= find(ismember(strhead(mnt.clab), strhead(dat.clab)));
dispChans= intersect(dispChans, find(~isnan(mnt.x)));
ax= zeros(1, length(dispChans));
hl= zeros(2, length(dispChans));

for icIdx= 1:length(dispChans),
  ic = dispChans(icIdx) ;
  ax(icIdx)= axes('position', getAxisHeadPos(mnt, ic, axisSize));
  if ~isempty(yLim),
    hc = imagesc(dat.t(timeIdx),dat.f(bandIdx),dat.x(bandIdx,timeIdx,icIdx),yLim) ;
  else
    hc = imagesc(dat.t(timeIdx),dat.f(bandIdx),dat.x(bandIdx,timeIdx,icIdx)) ;
  end ;
  set(ax(icIdx),'yDir','normal'); 
  hold on;
  if (dat.t(timeIdx(1))<0 & dat.t(timeIdx(end))>0),
    plot(0,dat.f(bandIdx(1)):.05:dat.f(bandIdx(end)),'k:');
  end ;
  
  axis off;
end

if isempty(yLim),
  yLim = [min(min(min(dat.x(bandIdx,timeIdx,:)))) max(max(max(dat.x(bandIdx,timeIdx,:))))];
end ;
yLimStr= sprintf('[%0.0f %0.0f] dB', trunc(yLim));

if isfield(dat, 't'),
  xLimStr= sprintf('[%0.0f %0.0f] ms ', trunc(dat.t(timeIdx([1 end]))));
else
  xLimStr= '';
end
if isfield(dat, 'f'),
  fLimStr= sprintf('[%0.1f %0.1f] Hz ', trunc(dat.f(bandIdx([1 end]))));
else
  fLimStr= '';
end

tit= sprintf(' %s %s %s', xLimStr, fLimStr ,yLimStr);
if isfield(dat, 'title'),
  fileSepPos = strfind(dat.title, filesep);
  if ~isempty(fileSepPos),
    tit= [untex(dat.title((fileSepPos(end)+1):end)) ':  ' tit];
  end ;
  tit= [untex(dat.title) ':  ' tit];
end
ht= addTitle(tit, 1);




