file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 150); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
scale= 1;
%classy= {'optimalCut', 'LSR'};
%classy= {'equalpriors', 'LSR'}
classy= 'linearPerceptron';
%classy= 'FisherDiscriminant'; scale= 70;
E= trunc(-1700:10:800);
%E= trunc(-1200:10:800);
tubePercent= [5 10 15];
testInd= {find(mrk.y(1,:)), find(mrk.y(2,:))};
nEvents= size(mrk.y, 2);

cnt.divTe{1}= num2cell((1:nEvents)',2);
cnt.divTr{1}= cell(nEvents, 1);
for k= 1:nEvents,
  cnt.divTr{1}{k}= setdiff(1:nEvents, k);  
end

%lr_jits= [0, -40, -80, -120];
lr_jits= [-40, -80, -120, -160];
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
outTraces= zeros(nShifts, nEvents);
testTube= zeros(nShifts, 3+2*length(tubePercent), nTestClasses);
for is= 1:nShifts, fprintf('\r%d  ', is);
  epo= makeSegments(cnt, mrk, [E(is)-1300 E(is)]);
  eval(proc);
  for ie= 1:nEvents,
    outTraces(is,ie)= scale * feval(applyFcn, C(ie), fv.x(:,ie));
  end
  for ic= 1:nTestClasses,
    testTube(is, :, ic)= fractileValues(outTraces(is, testInd{ic})', ...
                                        tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));

fprintf('results not saved so far\n');

return



labels= mrk.y;
save testTraces_lr outTraces E labels proc classy scale lr_jits

return


plotTubeNoFaceAlpha(testTube, E);
title(untex(cnt.title));
saveFigure('divers/cont_classification');
