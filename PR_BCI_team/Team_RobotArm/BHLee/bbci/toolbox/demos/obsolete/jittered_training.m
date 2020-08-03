file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
tov= -120;

%% unjittered
epo= makeEpochs(cnt, mrk, [-400 0]+tov);
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_jumpingMeans(fv, 8, 2);
doXvalidationPlus(fv, 'FisherDiscriminant', [10 10]);

%% jittered
jits= [-50 0 50];
epo= makeEpochs(cnt, mrk, [-400 0]+tov, jits);
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_jumpingMeans(fv, 8, 2);
fv.test_jits= [0];
doXvalidationPlus(fv, 'FisherDiscriminant', [10 10]);
