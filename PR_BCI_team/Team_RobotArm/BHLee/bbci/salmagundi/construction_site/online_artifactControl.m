file= 'VPgg_07_09_21/imag_arrowVPgg_cut50';

setup_bbci_bet_unstable;
global WINDOW
WINDOW= 1;

send_cnt(file);



%% other matlab shell


setup_bbci_bet_unstable;
opt= [];
opt.bv_host= 'localhost';
opt.fs= 100;
opt.whiskerperc= 10;
opt.whiskerlength= 3;
opt.clab= {'not','E*'};

state= acquire_bv(opt.fs, opt.bv_host);

dispclab= chanind(state.clab, opt.clab);
iClab= chanind(state.clab, scalpChannels);
iClab= iClab(find(ismember(iClab, dispclab)));

%% TODO: order frontal to occipital with scalpChannels
winlen= 100;
nChan= length(iClab);
X= zeros(0, nChan);
majorClab= chanind(state.clab(iClab), '*z');
minorClab= setdiff(1:nChan, majorClab);

clf;
set(gcf, 'Color',[1 1 1]);
global fig_char
set(gcf, 'KeyPressFcn','global fig_char; fig_char= [fig_char,double(get(gcbo,''CurrentCharacter''))];set(gcbo,''CurrentCharacter'','' '');');

nCols= 51;
cmap= [0 0 0; jet(nCols); 1 1 1];
colormap(cmap);
ha= axes('Position',[0.06 0.05 0.93 0.94]);
%hi= image(zeros(nChan, 4));
set(gca, 'TickLength',[0 0]);
  
last_nWin= NaN;
while ~ismember(27, fig_char),  %% do until ESC is pressed
  fig_char= [];
  [data,bn,mp,mt,md]= acquire_bv(state);
  X= cat(1, X, data(:,iClab));
  
  %% TODO: optional trialwise based on markers
  nWin= floor(size(X,1)/winlen);
  if ~isequal(nWin, last_nWin) & nWin>0,
    
    %% calculate matrix of short term variances
    Xw= reshape(X(1:nWin*winlen,:), [winlen nWin nChan]);
    Xv= squeeze(var(Xw, 1))';
    
    mi= min(Xv(:));
    peak= max(Xv(:));
    perc= percentiles(Xv(:), [0 100] + [1 -1]*opt.whiskerperc);
    thresh= perc(2) + opt.whiskerlength*diff(perc);
    ma= max(Xv(find(Xv < thresh)));
    Vint= 2 + floor(nCols*(Xv-mi)/(ma+1e-2-mi));
    Vdisp= ones([nChan+4 nWin+4]);
    Vdisp(3:end-2, 3:end-2)= Vint;
    %% TODO: mark critical channels / epochs
%    Vdisp(iClab+2, 1)= nCols+2;
%    Vdisp(iClab+2, end)= nCols+2;
%    Vdisp(1, rTrials+2)= nCols+2;
%    Vdisp(end, rTrials+2)= nCols+2;
    image([-1:size(Xv,2)+2], [1:nChan+4], Vdisp);
    axis_yticklabel(state.clab(iClab(minorClab)), 'ytick',2+minorClab, ...
                    'hpos', -0.01, 'color',0.9*[1 1 1], 'fontsize',7);
    axis_yticklabel(state.clab(iClab(majorClab)), 'ytick',2+majorClab, ...
                    'hpos', -0.01, 'color',[0 0 0], ...
                    'fontsize',10, 'fontweight','bold');
    xTick= get(gca, 'XTick');
    xTick= setdiff(xTick, 0);
    set(gca, 'XTick',xTick, 'YTick',[]);
    
%    set(hi, 'CData', Xvar', 'XData',[1 size(Xvar,1)]);
%    set(ha, 'XLim', 'XData',[1 size(Xvar,1)]);
    drawnow;
  end
end
