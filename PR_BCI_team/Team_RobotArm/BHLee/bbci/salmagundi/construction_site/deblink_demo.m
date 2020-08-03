file= 'Gabriel_00_09_05/selfpaced2sGabriel';
%file= 'Pavel_01_11_23/selfpaced2sPavel';   %% works bad for this file

[cnt, mrk, mnt]= loadProcessedEEG(file);
blk= getBlinkMarkers(cnt);

epo= makeSegments(cnt, blk, [-300 300]);
epo= proc_baseline(epo, [-300 -250]);
epo= proc_selectChannels(epo, chanind(epo, 'not', 'E*'));

iv= getIvalIndices([-20 20], epo);
blink_w= mean(mean(epo.x(iv,:,:)), 3)';
P_deblink= eye(length(blink_w)) - blink_w*blink_w'/(blink_w'*blink_w);

epo_deb= proc_linearDerivation(epo, P_deblink);
epo_deb.clab= epo.clab;
epo_deb.className= {'deblinked'};
Epo= proc_appendEpochs(epo, epo_deb);

showERPgrid(Epo, mnt);


fprintf('press <ret>\n'); pause

set(gca, 'colorOrder',[1 0 0; 0 0.9 0; 0.8 0 0; 0 0.7 0]);
cnt= proc_selectChannels(cnt, chanind(cnt, 'not', 'E*'));
erp= makeSegments(cnt, mrk, [-1200 600]);
erp= proc_baseline(erp, [-1200 -800]);
cnt_deb= proc_linearDerivation(cnt, P_deblink);
erp_deb= makeSegments(cnt_deb, mrk, [-1200 600]);
erp_deb= proc_baseline(erp_deb, [-1200 -800]);
erp_deb.className= {'left deblinked', 'right debinked'};
Erp= proc_appendEpochs(erp, erp_deb);

showERPgrid(Erp, mnt);
