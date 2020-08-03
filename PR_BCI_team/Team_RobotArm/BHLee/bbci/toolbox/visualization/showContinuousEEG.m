editfunction [hp, H]= showContinuousEEG(cnt, ival, mrk, varargin)
%hp= showContinuousEEG(cnt, ival, <mrk, opt>)
%
% IN  cnt         - structure of continuous EEG data
%     ival        - interval that should be displayed, [msec msec]
%     mrk         - marker structure
%     opt is a struct with fields
%     .scaleFactor- basic scaling factor is the median of 1/std(chans) devided
%                   by 10. Then given scalingFactor is multiplied with this
%                   basic scaling factor.
%     .scaling    - instead of using the median of 1/std(chans) you can 
%                   apply a fixed scaling here (default: [])
%     .col        - the colOrder as rgb values (default:[])

nChans= size(cnt.x,2);
opt= propertylist2struct(varargin{:});
opt = set_defaults(opt,'scaleFactor',1,...
                       'scaling',[],...
                       'scalePolicy', 'median', ...
                       'col',[], ...
                       'yLim',[0 nChans+1]);

scaleFactor = opt.scaleFactor;
col = opt.col;
scale = opt.scaling;

if exist('ival','var') & ~isempty(ival),
  cnt= proc_selectIval(cnt, ival);
else
  ival= [0 size(cnt.x,1)*1000/cnt.fs];
end

if ~isempty(scale) & length(scale)==1
  scale = scale*ones(1,nChans);
end


if isempty(col)
  col= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
end

cla;
hold on;
set(gca, 'colorOrder',col);

T= size(cnt.x,1);
baseline= nanmean(cnt.x(:,:), 1);
Base= ones(T,1) * baseline;
Shift= ones(T,1) * (1:nChans);
if isempty(scale)
  scale= 1./nanstd(cnt.x(:,:));
  %scale(find(isinf(scale)))= 1;
  scale(find(isinf(scale)))= 1;   %% exclude 'constant' channels from scaling
  scale(:)= feval(opt.scalePolicy, scale)/10 * scaleFactor;
end

eeg_traces= (Base - cnt.x(:,:))*diag(scale) + Shift;
time_line= linspace(ival(1)/1000, ival(2)/1000, T);
hp= plot(time_line, eeg_traces);
xLim= time_line([1 end]);
set(gca, 'xLim',xLim, ...
    'yLim',opt.yLim, 'yTick',1:nChans, 'yTickLabel',cnt.clab);
hzero= line(xLim'*ones(1,nChans), [1;1]*(1:nChans));
set(hzero, 'color','k', 'lineStyle',':');
hold off;
axis ij;
xlabel('[s]');

if exist('mrk', 'var') & ~isempty(mrk),
  if mrk.fs~=cnt.fs,
    error('marker and data sampling rate do not match');
  end
  ival_sa= ival/1000*cnt.fs;
  [so,si]= sort(mrk.pos);
  mrk_so= mrk_selectEvents(mrk, si);
  iShow= find(mrk_so.pos>=ival_sa(1) & mrk_so.pos<=ival_sa(2));
  H= showMarker(mrk_so, iShow, 1/cnt.fs, 0, opt);
end
H.zeroline= hzero;

if nargout==0,
  clear hp H;
end
