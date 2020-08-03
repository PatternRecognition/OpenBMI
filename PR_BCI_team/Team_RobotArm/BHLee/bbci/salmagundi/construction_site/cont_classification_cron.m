file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);

if JYM,
  spatFilt= {{'C3 lap', 'C3',1/2, 'C1',1/2, ...
                        'C5',-1/6, 'FC3',-1/6, 'FC1',-1/6, ...
                        'Cz',-1/6, 'CP3',-1/6, 'CP1',-1/6}, ...
             {'C4 lap', 'C4',1/2, 'C2',1/2, ...
                        'Cz',-1/6, 'FC2',-1/6, 'FC4',-1/6, ...
                        'C6',-1/6, 'CP2',-1/6, 'CP4',-1/6}, ...
             {'CP3 lap', 'CP3',1/2, 'CP1',1/2, ...
                         'CP5',-1/4, 'C3',-1/4, 'C1',-1/4, ...
                         'CPz',-1/4}, ...
             {'CP4 lap', 'CP4',1/2, 'CP2',1/2, ...
                         'CPz',-1/4, 'C2',-1/4, 'C4',-1/4, ...
                         'CP6',-1/4}};
  cnt= proc_spatialFilter(cnt, spatFilt);
  cnt= proc_baseline(cnt);
  proc= ['fv= proc_selectIval(epo, 740); ' ...
         'fv= proc_fourierBandReal(fv, [0.8 2.3], 128);'];
  scale= 1;
  classy= 'linearPerceptron';
  lr_jits= [-250, -180, -90, 0];
else        
  cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
  proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
         'fv= proc_jumpingMeans(fv, 5);'];
  %lr_jits= [-40, -80, -120, -160];
  lr_jits= [-50, -100, -150];
  scale= 1;
  classy= 'LSR';
  %classy= 'linearPerceptron';
  %classy= 'FisherDiscriminant'; scale= 70;
end
E= trunc(-1200:10:800);
testInd= {find(mrk.y(1,:)), find(mrk.y(2,:))};
tubePercent= [5 10 15];
nEvents= size(mrk.y, 2);
nTests= ceil(nEvents*1/4);

test_start= nEvents-nTests+1;   
%test_start=  ceil(nEvents*1/4);
test_idx= test_start-1 + (1:nTests);
train_idx= setdiff(1:nEvents, test_idx);

cnt.divTr{1}= {train_idx};
cnt.divTe{1}= {test_idx};

epo= makeSegments(cnt, mrk, [-1300 0], lr_jits);

eval(proc);
C= doXvalidationPlus(fv, classy, [], 'train only');
[func, params]= getFuncParam(classy);
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end
proc= [proc ', fv= proc_flaten(fv);'];
nShifts= length(E);
nTestClasses= length(testInd);
outTraces= zeros(nShifts, length(test_idx));
testTube= zeros(nShifts, 3+2*length(tubePercent), nTestClasses);
mrk_test= pickEvents(mrk, test_idx);
testInd= {find(mrk_test.y(1,:)), find(mrk_test.y(2,:))};
for is= 1:nShifts, fprintf('\r%d  ', is);
  epo= makeSegments(cnt, mrk_test, [E(is)-1300 E(is)]);
  eval(proc);
  outTraces(is,:)= scale * feval(applyFcn, C, fv.x);
  for ic= 1:nTestClasses,
    testTube(is, :, ic)= fractileValues(outTraces(is, testInd{ic})', ...
                                        tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));

fprintf('\nresults not saved so far\n');


return

labels= mrk.y(:,test_idx);
save cronTraces_lr outTraces E labels proc classy scale lr_jits


return

plotTubeNoFaceAlpha(testTube, E);
title(untex(cnt.title));
saveFigure('divers/cont_classification_cron');
