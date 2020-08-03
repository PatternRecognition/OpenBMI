function [hnd, w_chan, w_time]= plot_classifierImage(C, fv, varargin)
%[hnd, w_chan, w_time]= plot_classifierImage(C, fv)
%
% IN   C      - struct of trained linear hyperplane classifier
%      fv     - struct of feature vectors
%
% OUT  hnd    - struct with handle to graphic objects
%      w_chan - abs(classifer weights) averaged across time
%      w_time - abs(classifer weights) averaged across channels

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'show_title', 1, ...
                  'fontSize', 8);

if ~isfield(C,'w') | isfield(C,'sq'),
  error('only for linear classifiers');
end

nChans= length(fv.clab);
nDim= length(C.w)/nChans;
if isfield(fv,'t'),
  time_line= fv.t;
else
  time_line= 1:nDim;
end

ww= reshape(C.w, nDim, nChans);
w_chan= mean(abs(ww), 1);
w_time= mean(abs(ww), 2);

xp= 0.95;
xw= 0.03;
yp= 0.03;
yw= 0.05;

clf;
col= colormap(gray(60));
colormap(flipud(col([1:54 end],:)));

hnd.ax(1)= axes('position', [0.04 0.13 0.85 0.75]);
hnd.image(1)= imagesc(time_line, 1:nChans, abs(ww'));
set(gca, 'yTick',1:nChans, 'yTickLabel',fv.clab, 'yAxisLocation','right', ...
         'tickLength',[0 0], 'fontSize',opt.fontSize);
hnd.ax(2)= axes('position', [0.04 yp 0.85 yw]);
hnd.image(2)= imagesc(time_line, 1, w_time');
set(gca, 'xTick',[], 'yTick',[]);
hnd.ax(3)= axes('position', [xp 0.13 xw 0.75]);
hnd.image(3)= imagesc(1, 1:nChans, w_chan');
set(gca, 'xTick',[], 'yTick',[]);

if isfield(fv, 'title') & opt.show_title,
  [hnd.title, hnd.ax(4)]= addTitle(untex(fv.title), 0, 0);
else
  hnd.ax(4)= axes('position', [0 0 1 1]);
  set(hnd.ax(4), 'visible','off');
end
ht= text(xp+xw/2, yp, '\Sigma');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',14);
ht= text(xp-xw/2, yp, '\leftarrow');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',12);
ht= text(xp+xw/2, (yp+0.15)/2, '\uparrow');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',8);
