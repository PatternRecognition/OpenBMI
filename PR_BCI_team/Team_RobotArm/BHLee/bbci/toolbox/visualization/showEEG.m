function [hp, hl]= showEEG(data, ival, mrk, varargin)
%hp= showEEG(data, ival, <mrk, opt>)
%
% IN  data        - structure of continuous EEG data or of a single EEG epoch
%     ival        - interval that should be displayed, [msec msec]
%     mrk         - marker structure
%     opt is a struct with fields
%     .scaleFactor- basic scaling factor is the median of 1/std(chans) devided
%                   by 10. Then given scalingFactor is multiplied with this
%                   basic scaling factor.
%     .scaling    - instead of using the median of 1/std(chans) you can 
%                   apply a fixed scaling here (default: [])
%     .col        - the colOrder as rgb values (default:[])
 
nChans= size(data.x,2);
opt= propertylist2struct(varargin{:});
warning off
opt = set_defaults(opt,'scaleFactor',1,...
                       'scaling',[],...
                       'scalePolicy', 'median', ...
                       'baseline', 1, ...
                       'col',[], ...
                       'yLim',[0 nChans+1], ...
                       'CNT', 1, ...
                       'muVScale', 10);
warning on                   
                   
if ndims(data.x)==3
    if size(data.x, 3)==1
        data.x= squeeze(data.x);
    else
        error(['Too many epochs in variable ''data'' (single epoch expected, ', ...
               int2str(size(data.x,3)), ' epochs given.']);
    end
end       
    
scaleFactor = opt.scaleFactor;
col = opt.col;
scale = opt.scaling;

warning off
if exist('ival','var') && ~isempty(ival),
  data= proc_selectIval(data, ival);
else
  ival= [0 size(data.x,1)*1000/data.fs];
end
warning on

if ~isempty(scale) && length(scale)==1
  scale = scale*ones(1,nChans)*scaleFactor;
end

if isempty(col)
  col= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
end

cla;
hold on;
set(gca, 'colorOrder',col);

T= size(data.x,1);
if opt.baseline,
  baseline= nanmean(data.x(:,:), 1);
else
  baseline= zeros(1, nChans);
end
Base= ones(T,1) * baseline;
Shift= ones(T,1) * (1:nChans);  % vertial shift to visualize the channels
                                % at different y-positions
if isempty(scale)
  scale= 1./nanstd(data.x(:,:));
  scale(find(isinf(scale)))= 1;   %% exclude 'constant' channels from scaling
  scale(:)= feval(opt.scalePolicy, scale)/10 * scaleFactor;
end

if opt.CNT
    time_scale= 1000;  % sec-scale (x-axis) for continuous data
else
    time_scale= 1;     %  msec-scale (x-axis) for epoched data
end

eeg_traces= (Base - data.x)*diag(scale) + Shift;
time_line= linspace(ival(1)/time_scale, ival(2)/time_scale, T);
hp= plot(time_line, eeg_traces);
xLim= time_line([1 end]);

% plot muV-scale 
if ~isequal(opt.muVScale, 'off'),
  scl= scale(1);
  h= opt.muVScale*scl;      % default: 10muV scale 
  l= 0.25*h*(ival(2)/time_scale-ival(1)/time_scale)/(nChans+2);
  xPos= (ival(1)/time_scale)+0.1*(ival(2)/time_scale-ival(1)/time_scale);
  if nChans<=20
    for chan= 1:2:nChans     % plot a scale for half of the channels
                             % plot interval bar
      plot([xPos, xPos], [Shift(1,chan), (Shift(1,chan)-h)], ...
           'LineWidth', 1.5, 'Color', 'k');           % vertical bar
      plot([xPos-l, xPos+l], [(Shift(1,chan)-h), (Shift(1,chan)-h)], ...
           'LineWidth', 1.5, 'Color', 'k');           % horizont. bar
      plot([xPos-l, xPos+l], [Shift(1,chan), Shift(1,chan)], ...
           'LineWidth', 1.5, 'Color', 'k');           % horizont. bar
                                                      % plot label
      text(xPos+l, Shift(1,chan)-0.7*h, [int2str(opt.muVScale) '\muV'], ...
           'FontUnits','normalized', 'FontSize', min(0.02,max(0.017,h/40)), ...
           'FontWeight', 'bold');
%    fontsize= h/20
    end
  end
end

set(gca, 'xLim',xLim, ...
    'yLim',opt.yLim, 'yTick',1:nChans, 'yTickLabel',data.clab);
hl= line(xLim'*ones(1,nChans), [1;1]*(1:nChans));
set(hl, 'color','k', 'lineStyle',':');
hold off;
axis ij;
if opt.CNT
    xlabel('[s]');
else
    xlabel('[ms]');
end
    
if exist('mrk', 'var') && ~isempty(mrk),
  if mrk.fs~=data.fs,
    error('marker and data sampling rate do not match');
  end
  ival_sa= ival/1000*data.fs;
  [so,si]= sort(mrk.pos);
  mrk_so= mrk_selectEvents(mrk, si);
  iShow= find(mrk_so.pos>=ival_sa(1) & mrk_so.pos<=ival_sa(2));
  showMarker(mrk_so, iShow, 1/data.fs, 0);
end

if nargout==0,
  clear hp;
end
