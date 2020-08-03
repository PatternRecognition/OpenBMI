%setup_bbci_bet_braunschweig
%cd([BCI_DIR 'bbci_bet_braunschweig'])
%makeacquire_bv_braunschweig

fs= 1000;
lag= fs/100;

bs= acquire_bv_braunschweig(fs, 'brainamp')
[x, bn] = acquire_bv_braunschweig(bs);

[n,Wn]= buttord(30/fs*2, 50/fs*2, 3, 30);
[filt_b,filt_a]= butter(n, Wn);

T= 5;
TT= T*fs;
nChans= size(x,2);
xx= zeros(TT*3, nChans);
figure(1); clf;
set(gcf, 'DoubleBuffer','on');
hp= plot(xx(1:lag:TT,:));
cmap= cmap_rainbow(nChans-1);
cmap= cat(1, [0 0 0], cmap);
for cc= 1:nChans,
  set(hp(cc), 'Color',cmap(cc,:));
end
set(hp(1), 'LineWidth',2);
legend(bs.clab, -1);
set(gca,'XLim',[1 TT/lag], 'YLim',[-10 10]);
k= TT;
[dmy, filt_state]= filter(filt_b, filt_a, xx, [], 1);

while 1,
  [x, bn] = acquire_bv_braunschweig(bs);
  nSamples= size(x,1);
  [xf, filt_state]= filter(filt_b, filt_a, x, filt_state, 1);
  xx(k+[1:nSamples],:)= x; %% xf
  k= k+nSamples;
  if k>TT*2,
    xx(1:TT*2,:)= xx(TT+1:TT*3,:);
    k= k-TT;
  end
  for cc= 1:nChans,
    set(hp(cc), 'YData', xx(k-[TT:-lag:1],cc));
  end
  drawnow;
end
