function [ax] = showSpecgram(dat, chan, band, tLim, yLim)
% [ax] = showSpecgram(dat, chan, <band, tLim, yLim>)
%
% visualiszes the result of proc_specgram of a single channel
%
%   IN: dat    - special structure similar to epo or cnt (output of proc_specgram)
%       chan   - string containing the channel name or numerical
%                value with the index of the channel
%       band   - frequency range to display [Hz]
%       tLim   - time interval to display   [ms]
%       yLim   - amplitude range for the color-coding
%
%  OUT: ax     - axis handle
%
% in case of open questions read the source code or contact
%
% stl, Berlin Aug. 2004
%
% see also proc_specgram

[nFreq,nTime, nChan] = size(dat.x) ; 
   
if ~exist('yLim', 'var'), yLim=[]; end
if ~exist('tLim', 'var') | isempty(tLim), 
  timeIdx = 1: nTime;
else
  timeIdx = find(dat.t>=tLim(1) & dat.t<=tLim(end));
end ;
if ~exist('band', 'var') | isempty(band),
  bandIdx = 1: nFreq;  
else
  bandIdx = find(dat.f>=band(1) & dat.f<=band(end)) ;
end

clf;

icIdx = chanind(dat,chan) ;
icIdx = icIdx(1) ;
if ~isempty(yLim),
  hc = imagesc(dat.t(timeIdx),length(bandIdx),dat.x(bandIdx,timeIdx,icIdx),yLim) ;
else
  hc = imagesc(dat.t(timeIdx),dat.f(bandIdx),dat.x(bandIdx,timeIdx,icIdx)) ;
end ;
ax = gca ;
set(gca,'yDir','normal');
hold on;
if (dat.t(timeIdx(1))<0 & dat.t(timeIdx(end))>0),
  plot(0,dat.f(bandIdx(1)):.05:dat.f(bandIdx(end)),'k:');
end ;
  

if isempty(yLim),
  yLim = [min(min(dat.x(bandIdx,timeIdx,icIdx))) max(max(dat.x(bandIdx,timeIdx,icIdx)))];
end ;


if isfield(dat, 't'),
  xLimStr= sprintf('[%g %g] ms ', trunc(dat.t(timeIdx([1 end]))));
else
  xLimStr= '';
end
if isfield(dat, 'f'),
  fLimStr= sprintf('[%0.1f %0.1f] Hz ', trunc(dat.f(bandIdx([1 end]))));
else
  fLimStr= '';
end
title(dat.clab(icIdx));
yLimStr= sprintf('[%2.0f %2.0f] dB', trunc(yLim));
tit= sprintf(' %s %s %s', xLimStr, fLimStr ,yLimStr);
if isfield(dat, 'title'),
  fileSepPos = strfind(dat.title, filesep);
  if ~isempty(fileSepPos),
    tit= [untex(dat.title((fileSepPos(end)+1):end)) ':  ' tit];
  end ;
  tit= [untex(dat.title) ':  ' tit];
end
ht= addTitle(tit, 1);




