%% Steven's tdsep variant for extracting event-related components

file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
chEEG= chanind(cnt, 'not','E*');

epo= makeSegments(cnt, mrk, [-1200 600]);
ll= length(epo.t);
tau= ll*(0:5);
epox= permute(epo.x(:,chEEG,:), [2 1 3]);
epox= reshape(epox, [length(chEEG) ll*size(epo.x,3)]);
W= tdsep0(epox, tau);


%% show ERP of ICs
cnt_ic= copyStruct(cnt, 'x','clab');
cnt_ic.x= cnt.x(:, chEEG) * W';
for ci= 1:length(chEEG),
  cnt_ic.clab{ci}= sprintf('ic%0d', ci);
end
epo_ic= proc_baseline(epo_ic, [-1200 -800]);
mnt_ic= setElectrodeMontage(cnt_ic.clab); 
showERPgrid(epo_ic, mnt_ic, '-');

pause


%% show scalp patterns of ICs
A= inv(W);
nICs= size(W,1);
clf;
for ic= 1:nICs,
  suplot(nICs, ic, .01, .01);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range');
  hy= ylabel(sprintf('ic %d', ic));
  set(hy, 'visible','on');
end
