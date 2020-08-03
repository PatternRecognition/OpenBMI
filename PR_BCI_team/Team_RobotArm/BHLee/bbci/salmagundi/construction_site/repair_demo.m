file = 'Maik_02_04_24/selfpacedmulti2sMaik';

[cnt,mrk,mnt] = loadProcessedEEG(file);

epo= makeSegments(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -800]);
epo = proc_selectChannels(epo,'CCP3');
mnt= setDisplayMontage(mnt, sprintf('CCP3'));
showERPgrid(epo, mnt);                   


epo= makeSegments(cnt, mrk, [-1200 600]);
epo = proc_selectChannels(epo,'CCP3');
epo= proc_setMean(epo,{},struct('amplitude',[-2000 2000]));
epo= proc_baseline(epo, [-1200 -800]);

figure;showERPgrid(epo, mnt);                                                         

