file= 'Gabriel_00_09_05/selfpaced2sGabriel'; timeLimit= 2000;

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 5], 128, 150); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
classy= 'FisherDiscriminant';
xTrials= [1 1];                        %% leave-one-out
%xTrials= [5 10];
E= trunc(-800:10:800);
tubePercent= [5 10 15];
testInd= {find(mrk.y(1,:)), find(mrk.y(2,:))};

%% left vs right
[cnt.divTr, cnt.divTe] = sampleDivisions(mrk.y, xTrials);
lr_jits= [-40 -80 -120];
epo= makeSegments(cnt, mrk, [-1300 0], lr_jits);
epo.nJits= length(lr_jits)+1;
%nBase= length(epo.y)/epo.nJits;

eval(proc);
C= doXvalidationPlus(fv, classy, xTrials, 'train only');
[func, params]= getFuncParam(classy);
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end
proc= [proc ', fv= proc_flaten(fv);'];
nTrials= length(epo.divTe);
nShifts= length(E);
nEvents= size(mrk.y, 2);
nTestClasses= length(testInd);
outTraces= zeros(nTrials, nShifts, nEvents);
testTube= zeros(nShifts, 3+2*length(tubePercent), nTestClasses);
for is= 1:nShifts,
  epo= makeSegments(cnt, mrk, [E(is)-1300 E(is)]);
  eval(proc);
  for it= 1:nTrials,
    nDiv= length(epo.divTe{it});
    for id= 1:nDiv,
      n= id+(it-1)*nDiv;
      idxTe= epo.divTe{it}{id};
      out= feval(applyFcn, C(n), fv.x);
      outTraces(it, is, idxTe)= out(idxTe);
    end
  end
  for ic= 1:nTestClasses,
    outCl= outTraces(:, is, testInd{ic});
    testTube(is, :, ic)= fractileValues(outCl(:), tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));

save testTraces_lr outTraces E




return


file= 'Gabriel_00_09_05/selfpaced2sGabriel'; timeLimit= 2000;

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
classy= 'FisherDiscriminant';
xTrials= [5 10];
E= trunc(-800:10:800);
tubePercent= [5 10 15];

testInd= {find(any(mrk.y))};
classy= 'QDA';

%% movement vs no movement
motoJits= [-40, -80, -120 -160];
epo= makeSegments(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeSegments(cnt, mrk, [-1300 0]-800, [-60, -120, -180, -240]);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
epo.className= {'motor', 'no motor'};
epo.y= [repmat([1;0], 1, nMotos) repmat([0;1], 1, noMotos)];
[cnt.divTr, cnt.divTe] = sampleDivisions(epo.y, xTrials);

eval(proc);
C= doXvalidationPlus(fv, classy, xTrials, 'train only');
[func, params]= getFuncParam(classy);
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end
proc= [proc ', fv= proc_flaten(fv);'];
nTrials= length(epo.divTe);
nShifts= length(E);
nEvents= size(mrk.y, 2);
nTestClasses= length(testInd);
outTraces= zeros(nTrials, nShifts, nEvents);
testTube= zeros(nShifts, 3+2*length(tubePercent), nTestClasses);
for is= 1:nShifts,
  epo= makeSegments(cnt, mrk, [E(is)-1300 E(is)]);
  eval(proc);
  for it= 1:nTrials,
    nDiv= length(epo.divTe{it});
    for id= 1:nDiv,
      n= id+(it-1)*nDiv;
      idxTe= epo.divTe{it}{id};
      out= feval(applyFcn, C(n), fv.x);
      outTraces(it, is, idxTe)= out(idxTe);
    end
  end
  for ic= 1:nTestClasses,
    outCl= outTraces(:, is, testInd{ic});
    testTube(is, :, ic)= fractileValues(outCl(:), tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));
