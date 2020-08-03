file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
proc= ['fv= proc_filtBruteFFT(epo, [0.8 2.3], 128, 300); ' ...
       'fv= proc_jumpingMeans(fv, 5);'];
classy= 'LSR';
%classy= 'QDA';

%motoJits= [0, -40, -80, -120, -160];
motoJits= [-70, -100, -130, -160, -200];
%nomotoJits= [0, -60, -120, -180, -240]-1000;
nomotoJits= [-1000 -1100 -1200 550 750];
shift= 0;

E= trunc(-1200:10:800);
tubePercent= [5 10 15];
testInd= {find(any(mrk.y))};

%% movement vs no movement
epo= makeSegments(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeSegments(cnt, mrk, [-1300 0]-shift, nomotoJits);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
epo.className= {'motor', 'no motor'};
epo.y= [repmat([0;1], 1, nMotos) repmat([1;0], 1, noMotos)];
nEvents= size(mrk.y, 2);

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

cnt.divTe{1}= num2cell((1:nEvents)',2);
cnt.divTr{1}= cell(nEvents, 1);
for k= 1:nEvents,
  %% exclude test trial AND neighbouring no_movement events
  exclude= [k nEvents + [max(1,k-1) k+1]];
  cnt.divTr{1}{k}= setdiff(1:2*nEvents, exclude);  
end

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
outTraces= zeros(nShifts, nEvents);
testTube= zeros(nShifts, 3+2*length(tubePercent), nTestClasses);
for is= 1:nShifts,
  epo= makeSegments(cnt, mrk, [E(is)-1300 E(is)]);
  eval(proc);
  for ie= 1:nEvents,
    outTraces(is,ie)= feval(applyFcn, C(ie), fv.x(:,ie));
  end
  for ic= 1:nTestClasses,
    testTube(is, :, ic)= fractileValues(outTraces(is, testInd{ic})', ...
                                        tubePercent);
  end
end

plotTube(testTube, E);
title(untex(cnt.title));

save testTraces_mn outTraces
