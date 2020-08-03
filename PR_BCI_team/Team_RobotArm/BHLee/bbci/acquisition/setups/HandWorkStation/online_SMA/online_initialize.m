
function wld = online_initialize(wld)
%
% IN: T_epo - length of epo (ms)
%  wlen_sec - length of window for linfit analysis (ms)
% force_ival_sec - length of window for forced speed change condition
%         T - length of recording (sec)
%  strategy - 'linfit' or 'ranksum'
%   control - true/false


% resolution dependent parameters
wld.dt = wld.T_epo/1000;            % resolution of cfy output (sec)
wld.wlen = round(wld.wlen_sec/wld.dt);  % wlen -> sec
wld.nn = wld.T/wld.dt;              % number of data points


% initialize state
if strcmp(wld.strategy,'linfit')
    wld.y_lp = zeros(1,wld.nn-wld.wlen+1);
    wld.thresh = .1;
    wld.fig.ylim2 = [-(wld.wlen-1) wld.wlen-1];
elseif strcmp(wld.strategy,'ranksum')
    wld.dead_time = round(wld.wlen*2/3);
    wld.state.dead_time = 0;
    wld.alpha = .05;
    wld.force_ival = round(wld.force_ival_sec/wld.dt);
    wld.fig.ylim2 = [-(-log(wld.alpha)) -log(wld.alpha)];
else
    error('Unknown workload detection strategy.')
end
wld.state.n_hi = 0;
wld.state.n_lo = 0;
wld.state.hi = false;
wld.state.lo = false;


% figure
paper_width  = 35;
paper_heigth = 15;
wld.fig.h = figure('color','white',...
                  'paperunits','centimeters',...
                  'papersize',[paper_width paper_heigth],...
                  'paperposition',[0 0 paper_width paper_heigth]);
set(wld.fig.h,'units','centimeters','pos',[1,1,paper_width,paper_heigth])

wld.fig.induced = 0;
wld.fig.forced = 0;
wld.fig.ylim = [-1 1];

wld.fig.ax_state = subplot(6,8,1:8:6*8);
set(wld.fig.ax_state,'xlim',[-.5 .5],'ylim',wld.fig.ylim2,'ytick',[],'xtick',[],'box','on')
set(get(wld.fig.ax_state,'ylabel'),'string','detector state')

wld.fig.ax_Cout = subplot(6,8,[2:8 10:16 18:24]);
set(wld.fig.ax_Cout,'xlim',[wld.dt wld.T],'ylim',wld.fig.ylim,'ytick',[],'box','on','yaxislocation','right')
set(get(wld.fig.ax_Cout,'ylabel'),'string','Classifier output')

wld.fig.ax_induced = subplot(6,8,26:32);
set(wld.fig.ax_induced,'xlim',[wld.dt wld.T],'ylim',[0 1],'ytick',[],'xtick',[],'box','on','yaxislocation','right')
set(get(wld.fig.ax_induced,'ylabel'),'string','induced/forced')

wld.fig.ax_detected = subplot(6,8,34:40);
set(wld.fig.ax_detected,'xlim',[wld.dt wld.T],'ylim',[0 1],'ytick',[],'xtick',[],'box','on','yaxislocation','right')
set(get(wld.fig.ax_detected,'ylabel'),'string','detected')

wld.fig.ax_speed = subplot(6,8,42:48);
set(wld.fig.ax_speed,'xlim',[wld.dt wld.T],'ylim',[wld.speed.minmax(1)-1 wld.speed.minmax(2)+1],...
                     'xtick',[],'box','on','yaxislocation','right',...
                     'ytick',[wld.speed.minmax(1) sum(wld.speed.minmax)/2 wld.speed.minmax(2)])
set(get(wld.fig.ax_speed,'ylabel'),'string','speed')



