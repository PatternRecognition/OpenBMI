file= 'Gabriel_00_10_04/selfpacedmultiGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file, 'cut50');
epo= makeSegments(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -800]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
showERPgrid(epo, mnt);
pause

epo_lap= proc_laplace(epo, 'small');
mnt_lap= adaptMontage(mnt, epo_lap, ' lap');
showERPgrid(epo_lap, mnt_lap);
pause

showERtvalueGrid(epo_lap, mnt_lap);
pause

%% low-level
iv= getIvalIndices([-70 -20], epo);
left= find(epo.y(1,:));
right= find(epo.y(2,:));
both= find(epo.y(3,:));
bp_left= mean(mean(epo.x(iv,:,left)),3);
bp_right= mean(mean(epo.x(iv,:,right)),3);
bp_both= mean(mean(epo.x(iv,:,both)),3);
subplot(131);
h= showScalpPattern(mnt, bp_left, 0);
title('BP for left events');
subplot(132);
h(2)= showScalpPattern(mnt, bp_right, 0);
title('BP for right events');
subplot(133);
h(3)= showScalpPattern(mnt, bp_both, 0);
title('BP for left+right events');
unifyCLim(h, 1);
