file= 'Gabriel_01_12_12/selfpaced0_5sGabriel'; timeLimit= 1000;
%fig_file= 'divers/emg_impact/eeg_vs_emg'

[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= pickEvents(mrk, 1500);

classy= 'FisherDiscriminant';
xTrials= [10 10];
E= trunc(-600:10:100);
equi.idcs= getEventPairs(mrk, timeLimit);
[cnt.divTr, cnt.divTe] = sampleDivisions(mrk.y, xTrials, equi);

cnt_eeg= proc_selectChannels(cnt, 'FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6');
errEEG= zeros(length(E), 2);
k= 0;
for lp= E,
  k= k+1;
  epo= makeSegments(cnt_eeg, mrk, [lp-1300 lp]);
  fv= proc_filtBruteFFT(epo, [0.8 2], 128, 150);
  fv= proc_jumpingMeans(fv, 5);
  fprintf('%5d> ', lp);
  errEEG(k,:)= doXvalidationPlus(fv, classy, [], 1);
end
clear cnt_eeg

cnt_emg= proc_selectChannels(cnt, 'EMG*');
errEMG= zeros(length(E), 2);
k= 0;
for lp= E,
  k= k+1;
  epo= makeSegments(cnt_emg, mrk, [lp-300 lp]);
  fv= proc_detectEMG(epo);
  fprintf('%5d> ', lp);
  errEMG(k,:)= doXvalidationPlus(fv, classy, [], 1);
end
clear cnt_emg

plot(E, [errEEG(:,1) errEMG(:,1)]);
legend('EEG', 'EMG');

if exist('fig_file', 'var'),
  saveFigure(fig_file);
end
