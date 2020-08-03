file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 2.3], 128, 300); ' ...
       'fv= proc_jumpingMeans(fv, 6);'];
%proc= ['fv= proc_selectIval(epo, 240);' ...
%       'fv= proc_subsampleByMean(fv, 8);'];
scale= 1;
%classy= 'linearPerceptron';
classy= 'LSR';
%classy= {'two_via_three', 'LSR'};

%motoJits= [0, -40, -80, -120, -160];
%motoJits= [-100, -150, -200];
motoJits= [-70, -100, -130, -160, -200];
%nomotoJits= [0, -60, -120, -180, -240]-1000;
%nomotoJits= [-1000, -1100, 600];
nomotoJits= [-1000, -1100, -1200, 550, 750];
shift= 0;

nEvents= size(mrk.y, 2);
nTests= ceil(nEvents*1/4);

test_start= nEvents-nTests+1;   
%test_start=  ceil(nEvents*1/4);
test_idx= test_start-1 + (1:nTests);
train_idx= setdiff(1:nEvents, test_idx);

E= trunc(-1200:10:800);
tubePercent= [5 10 15];

%% movement vs no movement
epo= makeSegments(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeSegments(cnt, mrk, [-1300 0]-shift, nomotoJits);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
epo.y= zeros(3, size(epo.y,2));
%% labels for no motor events
epo.y(1,nMotos+1:end)= 1;
%% labels for l/r motor events
epo.y(2:3,1:nMotos)= repmat(mrk.y, [1 length(motoJits)]);  

%% resort events for jittered training
if length(motoJits)~=length(nomotoJits),
  error('sorry, this script works only for equal number of jitters');
end
nJits= length(motoJits);
epo.nJits= nJits;
idx= [];
for jj= 1:nJits,
  idx= [idx (1:nEvents)+(jj-1)*nEvents (1:nEvents)+(jj-1+nJits)*nEvents];
end
epo.x= epo.x(:,:,idx);
epo.y= epo.y(:,idx);

cnt.divTe{1}= {test_idx};
%% for training take all movements of the training period and
%% and in-beween 'no movement' periods, except for the first
%% and last one, because they might be neighbouring to some
%% test trails
cnt.divTr{1}= {[train_idx, nEvents+train_idx(2:end-1)]};

mrk_test= pickEvents(mrk, test_idx);
testInd= {find(any(mrk_test.y))};
epo.divTr= cnt.divTr;
epo.divTe= cnt.divTe;
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

for is= 1:nShifts, fprintf('\r%d  ', is),
  epo= makeSegments(cnt, mrk_test, [E(is)-1300 E(is)]);
  eval(proc);
  out3c= scale * feval(applyFcn, C, fv.x);
  if size(out3c,1)==3,
    outTraces(is,:)= [max(out3c(2:end,:))-out3c(1,:)];
  else
    outTraces(is,:)= out3c;
  end
  for ic= 1:nTestClasses,
    testTube(is, :, ic)= fractileValues(outTraces(is, testInd{ic})', ...
                                        tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));

fprintf('\nresults not saved so far\n');


return

save cronTraces_mn3c E outTraces proc classy scale motoJits nomotoJits


return

plotTubeNoFaceAlpha(testTube, E);
title(untex(cnt.title));
saveFigure('divers/cont_detection_cron');
