file= 'Gabriel_01_12_12/selfpaced0_5sGabriel'; timeLimit= 1000;
%fig_file= 'divers/emg_impact/good_vs_bad_equi'

[cnt, Mrk, mnt]= loadProcessedEEG(file);
pairs= getEventPairs(Mrk, timeLimit);
equi= equiSubset(pairs);
[mrk,ev]= pickEvents(Mrk, [equi{:}]);
for ic= 1:4,
  eq_pair{ic}= find(ismember(ev, pairs{ic}));
end

xTrials= [10 10];
E= trunc(-350:10:0);

cnt_eeg= proc_selectChannels(cnt, 'FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6');
cnt_emg= proc_selectChannels(cnt, 'EMG*');
left= find(mrk.y(1,:));
right= find(mrk.y(2,:));
nShifts= length(E);
nEvents= sum(mrk.y(:));
errEMG= zeros(nShifts, 2);
errEMG= zeros(nShifts, 2);
errEMGog= zeros(nShifts, 2);
errEMGob= zeros(nShifts, 2);
errEEGog= zeros(nShifts, 2);
errEEGob= zeros(nShifts, 2);
nGood= zeros(nShifts, 1);
nBad= zeros(nShifts, 1);

for is= 1:nShifts,
  epo= makeSegments(cnt_emg, mrk, E(is) + [-300 0]);
  fv_emg= proc_detectEMG(epo);
  
  C= train_FisherDiscriminant(fv_emg.x, fv_emg.y);
  out= C.w'*fv_emg.x+C.b;
  
  [lo,li]= sort(out(left));           %% retrain on best 20%
  goodLeft= left(li(1:round(0.2*length(left))));
  [ro,ri]= sort(out(right));
  goodRight= right(ri(end-round(0.2*length(right)):end));
  fv= setGoal(fv_emg, {goodLeft, goodRight});
  C= train_FisherDiscriminant(fv.x, fv.y);
  out= C.w'*fv_emg.x+C.b;
  err= sign(out) ~= [-1 1]*fv_emg.y;

  minEventsPerClass= 100;
  frac= 1-2*mean(err);
  nSub= max(minEventsPerClass, round(frac*length(err)/2));
  nSub= min(nSub, min(length(left),length(right))-minEventsPerClass);
  [lo,li]= sort(out(left));
  goodLeft= left(li(1:nSub));
  [ro,ri]= sort(out(right));
  goodRight= right(ri(end-nSub+1:end));
  goodEMG= [goodLeft goodRight];
  badLeft= left(li(nSub+1:end));
  badRight= right(ri(1:end-nSub));
  badEMG= [badLeft badRight];

  epo= makeSegments(cnt_eeg, mrk, E(is) + [-1300 0]);
  fv_eeg= proc_filtBruteFFT(epo, [0.8 2], 128, 150);
  fv_eeg= proc_jumpingMeans(fv_eeg, 5);

  yog= setGoal(mrk.y, {goodLeft, goodRight});
  yob= setGoal(mrk.y, {badLeft, badRight});
  nGood(is)= sum(yog(:));
  nBad(is)= sum(yob(:));

  fprintf('emg_on_all@%5d> ', E(is));
  fv_emg.y= mrk.y;
  fv_emg.equi.idcs= eq_pair;
  errEMG(is,:)= doXvalidationPlus(fv_emg, 'FisherDiscriminant', xTrials, 1);
  fprintf('emg_on_gut@%5d> ', E(is));
  fv_emg.y= yog;
  fv_emg.equi.idcs= metaintersect(eq_pair, goodEMG);
  errEMGog(is,:)= doXvalidationPlus(fv_emg, 'FisherDiscriminant', xTrials, 1);
  fprintf('emg_on_bad@%5d> ', E(is));
  fv_emg.y= yob;
  fv_emg.equi.idcs= metaintersect(eq_pair, badEMG);
  errEMGob(is,:)= doXvalidationPlus(fv_emg, 'FisherDiscriminant', xTrials, 1);

  fprintf('eeg_on_all[%4d]> ', nEvents);
  fv_eeg.y= mrk.y;
  fv_eeg.equi.idcs= eq_pair;
  errEEG(is,:)= doXvalidationPlus(fv_eeg, 'FisherDiscriminant', xTrials, 1);
  fprintf('eeg_on_gut[%4d]> ', nGood(is));
  fv_eeg.y= yog;
  fv_eeg.equi.idcs= metaintersect(eq_pair, goodEMG);
  errEEGog(is,:)= doXvalidationPlus(fv_eeg, 'FisherDiscriminant', xTrials, 1);
  fprintf('eeg_on_bad[%4d]> ', nBad(is));
  fv_eeg.y= yob;
  fv_eeg.equi.idcs= metaintersect(eq_pair, badEMG);
  errEEGob(is,:)= doXvalidationPlus(fv_eeg, 'FisherDiscriminant', xTrials, 1);

end
%clear cnt_emg cnt_eeg


save eeg_vs_emg_equi eq_pair E errEEG errEEGog errEEGob errEMG errEMGog errEMGob nGood nBad

plot(E, errEEG(:,1), 'b');
hold on;
plot(E, errEEGog(:,1), 'b-^');
plot(E, errEEGob(:,1), 'b-v');
plot(E, errEMG(:,1), 'r');
plot(E, errEMGog(:,1), 'r-^');
plot(E, errEMGob(:,1), 'r-v');
hold off;
legend('EEG on all', 'EEG on good', 'EEG on bad', 'EMG');
ylabel('validation error [%]');

if exist('fig_file', 'var'),
  addTitle([untex(cnt.title) ...
     ': classification on equilibrated ''good''- and ''bad''-EMG subsets']);
  saveFigure(fig_file);
end
