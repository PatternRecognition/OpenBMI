file= 'Gabriel_01_10_15/imagGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [0 1500]);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, [200 1000]);
fv= proc_jumpingMeans(fv, 40);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);


fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, [200 1000]);
fv= proc_arCoefs(fv, 10);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);
