file= 'Gabriel_00_10_04/selfpacedmultiGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-120);

%% do some preprocessing
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);


doXvalidation(fv, 'LDA', [10 10]);

doXvalidation(fv, 'FisherDiscriminant', [3 10]);
doXvalidation(fv, {'optimalCut', 'FisherDiscriminant'}, [3 10]);
doXvalidation(fv, {'optimalCutSVM', 'FisherDiscriminant'}, [3 10]);
%% haeh?



[cnt, mrk, mnt]= loadProcessedEEG(file);
jits= [20 40 60];
epo= makeSegments(cnt, mrk, [-1300 0]-120, jits);
epo.nJits= length(jits)+1;

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);


doXvalidationPlus(fv, 'LDA', [5 10]);
