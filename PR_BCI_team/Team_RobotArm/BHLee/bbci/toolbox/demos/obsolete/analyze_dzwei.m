file= 'Stefan_01_05_10/dzweiStefan';
%file= 'Gabriel_00_11_03/dzweiGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file, 'cut50');
epo= makeSegments(cnt, mrk, [-1000 500]);
epo= proc_baseline(epo, [-1000 -800]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
epo= proc_baseline(epo, [-1000 -800]);
showERPgrid(epo, mnt);
pause

epo_lap= proc_laplace(epo, 'small');
mnt_lap= adaptMontage(mnt, epo_lap, ' lap');
mnt_lap.box(:,end)= [0; -1];
showERPgrid(epo_lap, mnt_lap);
pause

[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= getDzweiEvents(file);
epo= makeSegments(cnt, mrk, [-800 800]);
epo= proc_baseline(epo, [-800 -600]);
epo= proc_rectifyChannels(epo, 'EMGl', 'EMGr');
epo= proc_baseline(epo, [-800 -600]);
showERPgrid(epo, mnt);
sum(epo.y,2)' 

fv= epo;
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_selectChannels(fv, 'FCz','Cz','CPz');
fv= proc_selectIval(fv, [-50 200]);
classy= 'gaussianERPmodel';
doXvalidation(fv, classy, [10 10]);
%% see also error_detection.m