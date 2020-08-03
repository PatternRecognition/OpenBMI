file= 'Gabriel_01_12_12/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-120);

%fv= proc_selectChannels(epo, 'FC5-6', 'CFC5-6', 'C5-6', 'CCP5-6', 'CP5-6');
fv= proc_selectChannels(epo, 'FC5-6', 'C5-6', 'CCP5-6', 'CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 2], 128, 150);
fv= proc_jumpingMeans(fv, 5, 3);

classy= 'FisherDiscriminant',
doXvalidation(fv, classy, [5 10]);


[d, mrk]= loadProcessedEEG(file, '', 'mrk');

classDef= {[74], [192]; 'right index', 'right little'};
%classDef= {[70], [65]; 'left index', 'left little'};
mrk= makeClassMarkers(mrk, classDef, 1000);
epo= makeSegments(cnt, mrk, [-1300 0]-120);

fv= proc_selectChannels(epo, 'FC5-6', 'CFC5-6', 'C5-6', 'CCP5-6', 'CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_subsampleByMean(fv, 5);

classy= 'FisherDiscriminant',
doXvalidation(fv, classy, [1 10]);

classy= 'QDA',
doXvalidation(fv, classy, [1 10]);
