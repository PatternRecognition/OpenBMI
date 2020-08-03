file= 'Gabriel_00_09_05/selfpaced2sGabriel';
%file= 'Roman_01_12_13/selfpaced2sRoman';

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
bp_left= mean(mean(epo.x(iv,:,left)),3);
bp_right= mean(mean(epo.x(iv,:,right)),3);
subplot(121);
h= showScalpPattern(mnt, bp_left, 0, 'horiz', [], 1);
title('BP for left events');
subplot(122);
h(2)= showScalpPattern(mnt, bp_right, 0, 'horiz', [], 1);
title('BP for right events');
unifyCLim(h, 1);
