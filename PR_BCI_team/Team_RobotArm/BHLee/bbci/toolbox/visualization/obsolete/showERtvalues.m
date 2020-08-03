function [N,hp]= showERtvalues(epo, mnt, Chan, varargin)
%[N,hp]= showERtvalues(epo, mnt, chan, <FLAGS>)
%
% IN    epo     - struct of epoched signals, see makeSegments
%       mnt     - struct for electrode montage, see setElectrodeMontage
%       chan    - channel label or index
%       FLAGS   -
%                 'small'     setup for small axes
%
% OUT  N       - number of averaged events

% bb, GMD.FIRST.IDA 09/00

SMALL= ~isempty(strmatch('small', {varargin{:}}));


nClasses= size(epo.y, 1);
if nClasses~=2, error('only for two classes'); end
chan= chanind(epo, Chan);
if isempty(chan),
  error('channel not found'); 
end

T= length(epo.t);
N= zeros(nClasses, 1);
xm= zeros(T, nClasses);
xs= zeros(T, nClasses);
for cc= 1:nClasses,
  ei= find(epo.y(cc,:));
  N(cc)= length(ei);
  x= squeeze(epo.x(:, chan, ei));
  xm(:,cc)= mean(x, 2);
  xv(:,cc)= var(x')';
end

df= sum(N)-2;
sxd= sqrt( ((N(1)-1)*xv(:,1)+(N(2)-1)*xv(:,2)) / df  * ...
           (1/N(1)+1/N(2)) );
xt= (xm(:,1)-xm(:,2))./sxd;

hp= plot(epo.t, xt);

set(gca, 'xLim', [epo.t(1) epo.t(end)]);

if SMALL,
  set(gca, 'xTick', 0, 'xTickLabel', [], 'xGrid', 'on', ...
           'yTickLabel', [], 'yGrid', 'on');
else
  title(epo.clab{chan});
%  set(line([0 0], get(gca, 'yLim')), 'color', 'k', 'lineStyle', '-.');
end
