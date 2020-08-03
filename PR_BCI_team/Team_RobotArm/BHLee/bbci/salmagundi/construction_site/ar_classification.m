file= 'Steven_01_06_13/dzweiSteven';
%file= 'Steven_01_11_20/selfpaced2sSteven';
%file= 'Steven_01_11_20/selfpaced5sSteven';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-50);

fprintf('[classification on ar coefficients]\n');
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, 400);
fv= proc_arCoefs(fv, 10);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);



fprintf('[classification on Bereitschaftspotenzial]\n');
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);
