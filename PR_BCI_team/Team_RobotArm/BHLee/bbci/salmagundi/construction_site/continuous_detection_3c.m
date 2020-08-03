file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
xTrials= [5 10];
E= trunc(-800:10:800);
tubePercent= [5 10 15];

testInd= {find(any(mrk.y))};
%classy= 'QDA';
classy= 'LSR';

%% movement vs no movement
motoJits= [-40, -80, -120 -160];
epo= makeSegments(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
nomotoJits= [-60, -120, -180, -240];
no_moto= makeSegments(cnt, mrk, [-1300 0]-800, nomotoJits);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
epo.nJits= length(motoJits) + length(nomotoJits) + 2;
clear no_moto
epo.y= zeros(3, size(epo.y,2));                 %% labels for l/r motor events
epo.y(1:2,1:nMotos)= repmat(mrk.y, [1 length(motoJits)+1]);  
epo.y(3,nMotos+1:end)= 1;                       %% labels for no motor events
epo.className= {'left','right','none'};

[cnt.divTr, cnt.divTe] = sampleDivisions(mrk.y, xTrials);
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
nEvents= size(epo.y, 2);
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
      out3c= feval(applyFcn, C(n), fv.x);
      out= [max(out3c(1:2,:))-out3c(3,:)];
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
