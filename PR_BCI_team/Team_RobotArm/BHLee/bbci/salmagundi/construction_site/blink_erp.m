file= 'Gabriel_00_09_05/selfpaced2sGabriel';
%file= 'Pavel_01_11_23/selfpaced2sPavel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
blk= getBlinkMarkers(cnt);

epo= makeSegments(cnt, blk, [-300 300]);
epo= proc_baseline(epo, [-300 -250]);

showERPgrid(epo, mnt);


fprintf('press <ret>\n'); pause

eegChans= setdiff(1:length(epo.clab), chanind(epo, 'E*'));
epo= proc_selectChannels(epo, eegChans);
clf;
iv= getIvalIndices([-20 20], epo);
blink_w= mean(mean(epo.x(iv,:,:)), 3)';
showScalpPattern(mnt, blink_w, [], [], 'range');
title(['blink gradient for ' untex(epo.title)]);
