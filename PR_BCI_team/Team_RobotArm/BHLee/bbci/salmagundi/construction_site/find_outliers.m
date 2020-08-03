file= 'Steven_01_06_13/dzweiSteven';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0]-50);

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_subsampleByMean(fv, 5);

[em,es,out,avErr,evErr]= doXvalidationPlus(fv, 'FisherDiscriminant', [10 10]);
fprintf('%.1f%% were always classified correctly\n', ...
        100*sum(evErr==0)/length(evErr));
fprintf('%.1f%% were never classified correctly\n', ...
        100*sum(evErr==1)/length(evErr));
outliers= find(evErr>0.1);
fprintf('%.1f%% were misclassified in > 10%% of the validation trials:\n', ...
        100*length(outliers)/length(evErr));
outliers
