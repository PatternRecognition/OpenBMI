file= 'Stefan_01_05_10/dzweiStefan';
%file= 'Gabriel_00_11_03/dzweiGabriel';
%file= 'Seppel_01_05_23/dzweiSeppel';
%file= 'Frank_01_07_19/dzweiFrank';
%file= 'Motoaki_01_06_07/dzweiMotoaki';
%file= 'Steven_01_06_13/dzweiSteven';
%file= 'Thorsten_01_05_15/dzweiThorsten';
%file= 'Ben_01_05_31/dzweiBen';

[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= getDzweiEvents(file);
epo= makeSegments(cnt, mrk, [-100 800]);

fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4');
fv= proc_selectIval(fv, [0 350]);
fv= proc_jumpingMeans(fv, 5);

nTrials= length(mrk.y);
nHits= sum(mrk.y(1,:));
nMisses= sum(mrk.y(2,:));
reac= mean(mrk.reac);
fprintf('%s: %d ms reaction time, %d errors in %d trials (%3.1f%%)\n\n', ...
        cnt.title, round(reac), nMisses, nTrials, 100*nMisses/nTrials);


fprintf('overall detection error\n');
doXvalidation(fv, 'FisherDiscriminant', [10 10]);

fp_bound= 0.02;

classy= {'boundErrorOfType1', fp_bound, 'FisherDiscriminant'};
fprintf('keep type-1-errors in training sets below %d%%.\n', 100*fp_bound);
[d,d, outTe]= doXvalidation(fv, classy, [20 5], 1);
[mc, me, ms]= calcConfusionMatrix(fv, outTe);
fprintf(['errors:  %.1f' 177 '%.1f%%,  %.1f' 177 '%.1f%%\n'], ...
        me(2), ms(2), me(3), ms(3));

fv.fp_bound= fp_bound;
fprintf('shift hyperplane to fix type-1-errors in test sets.\n');
[d,d, outTe]= doXvalidationPlus(fv, classy, [20 5], 1);
[mc, me, ms]= calcConfusionMatrix(fv, outTe);
fprintf(['errors:  %.1f' 177 '%.1f%%,  %.1f' 177 '%.1f%%\n'], ...
        me(2), ms(2), me(3), ms(3));
