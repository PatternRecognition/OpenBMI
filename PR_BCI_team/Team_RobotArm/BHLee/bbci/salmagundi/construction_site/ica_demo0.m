file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
chEEG= chanind(cnt, 'not','E*');

marker= readMarkerTable(file);
is= min(find(marker.toe==252));  %% start of first part
pos_s= marker.pos(is);
ie= min(find(marker.toe==253));  %% end of first part
pos_e= marker.pos(ie);

W= tdsep0(cnt.x(pos_s:pos_e, chEEG)', 0:3);


%% show ERP of ICs
cnt_ic= copyStruct(cnt, 'x','clab');
cnt_ic.x= cnt.x(:, chEEG) * W';
for ci= 1:length(chEEG),
  cnt_ic.clab{ci}= sprintf('ic%0d', ci);
end
epo_ic= makeSegments(cnt_ic, mrk, [-1200 600]);
epo_ic= proc_baseline(epo_ic, [-1200 -800]);
mnt_ic= setElectrodeMontage(cnt_ic.clab); 
showERPgrid(epo_ic, mnt_ic, '-');

pause


%% show scalp patterns of ICs
A= inv(W);
nICs= size(W,1);
for ic= 1:nICs,
  suplot(nICs, ic, .01, .01);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range');
  ylabel(sprintf('ic %d', ic));
end



return

%% pca

C= cov(cnt.x(pos_s:pos_e, chEEG));
[E,D]= eig(C);

A= inv(E);
for ic= 1:length(chEEG),
  suplot(length(chEEG), ic, .01, .01);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range');
  ylabel(sprintf('pc %d', ic));
end
