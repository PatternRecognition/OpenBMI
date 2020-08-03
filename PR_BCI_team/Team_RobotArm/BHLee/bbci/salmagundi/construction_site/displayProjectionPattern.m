function ha= displayProjectionPattern(fv, w)
%ha= displayProjectionPattern(fv, w)

[T, nChans, nEvents]= size(fv.x);
ww= reshape(w, [T, nChans]);

clf;
col= colormap('gray');
colormap(flipud(col([1:54 end],:)));
%colormap(flipud(col));

xp= 0.96;
xw= 0.03;
yp= 0.03; %% 0.9
yw= 0.05;
ha= axes('position', [0.01 0.13 0.9 0.8]);
imagesc(fv.t, 1:nChans, abs(ww'));
set(gca, 'yTick',1:nChans, 'yTickLabel',fv.clab, 'yAxisLocation','right', ...
         'tickLength',[0 0], 'fontSize',8);
axes('position', [0.01 yp 0.9 yw]);
imagesc(fv.t, 1, mean(abs(ww'), 1));
set(gca, 'xTick',[], 'yTick',[]);
axes('position', [xp 0.13 xw 0.8]);
imagesc(1, 1:nChans, mean(abs(ww'), 2));
set(gca, 'xTick',[], 'yTick',[]);

ax= axes('position', [0 0 1 1]);
set(ax, 'visible','off');
ht= text(xp+xw/2, yp, '\Sigma');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',14);
ht= text((0.9+0.01+(xp+xw/2))/2, yp, '\leftarrow');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',12);
ht= text(xp+xw/2, (yp+0.13)/2, '\uparrow');
set(ht, 'horizontalAlignment','center', ...
        'verticalAlignment', 'baseline', 'fontSize',8);

axes(ha);
