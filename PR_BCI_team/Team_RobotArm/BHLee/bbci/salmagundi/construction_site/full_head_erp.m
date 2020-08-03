file= 'Gabriel_01_12_12/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file, 'cut50');

epo= makeSegments(cnt, mrk, [-1200 600]);
epo= proc_baseline(epo, [-1200 -800]);

showERPhead(epo, mnt, [-12 12]);
