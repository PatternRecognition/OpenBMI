file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
classy= 'FisherDiscriminant';
xTrials= [1 1];                         %% leave-one-out
%xTrials= [5 10];
E= trunc(-800:10:800);
tubePercent= [5 10 15];

testInd= {find(any(mrk.y))};
classy= 'QDA';

%% movement vs no movement
motoJits= [-40, -80, -120 -160];
epo= makeSegments(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
nomotoJits= [-60, -120, -180, -240];
no_moto= makeSegments(cnt, mrk, [-1300 0]-800, nomotoJits);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
epo.className= {'motor', 'no motor'};
epo.y= [repmat([1;0], 1, nMotos) repmat([0;1], 1, noMotos)];
nEvents= size(mrk.y, 2);

%% resort events for jittered training
if length(motoJits)~=length(nomotoJits),
  error('sorry, this script works only for equal number of jitters');
end
nJits= length(motoJits)+1;
epo.nJits= nJits;
idx= [];
for jj= 1:nJits,
  idx= [idx (1:nEvents)+(jj-1)*nEvents (1:nEvents)+(jj-1+nJits)*nEvents];
end
epo.x= epo.x(:,:,idx);
epo.y= epo.y(:,idx);

[cnt.divTr, cnt.divTe] = sampleDivisions(epo.y(:,1:2*nEvents), xTrials);
epo.divTr= cnt.divTr;
epo.divTe= cnt.divTe;
eval(proc);
C= doXvalidationPlus(fv, classy, xTrials, 'train only');
[func, params]= getFuncParam(classy);
applyFcn= ['apply_' func];
if ~exist(applyFcn, 'file'),
  applyFcn= 'apply_separatingHyperplane';
end
proc= [proc ', fv= proc_flaten(fv);'];
nTrials= length(cnt.divTe);
nShifts= length(E);
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
      idxTe= intersect(epo.divTe{it}{id}, 1:nEvents);
      if ~isempty(idxTe),                           %% movement in test set?
        out= feval(applyFcn, C(n), fv.x);
        outTraces(it, is, idxTe)= out(idxTe);
      end
    end
  end
  for ic= 1:nTestClasses,
    outCl= outTraces(:, is, testInd{ic});
    testTube(is, :, ic)= fractileValues(outCl(:), tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));
