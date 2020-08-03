file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'FC#', 'C#', 'CP#');
clLen= 800;
proc= ['fv= proc_laplace(epo, ''small'', '' lap'', ''filter all''); ' ...
       'fv= proc_selectChannels(fv, ''C3'',''Cz'',''C4'',' ...
                                   '''CP3'',''CPz'',''CP4''); ' ...
       'fv= proc_fourierBandReal(fv, [0.8 2.3], 128);'];
classy= 'linearPerceptron';

motoJits= [-300 -200 -100 0];
shift= 0;
nomotoJits= [600 800 1000 1200];

E= trunc(-1200:10:600);
tubePercent= [5 10 15];
testInd= {find(any(mrk.y))};

%% movement vs no movement
epo= makeSegments(cnt, mrk, [-clLen 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeSegments(cnt, mrk, [-clLen 0]-shift, nomotoJits);
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
  exclude= [k nEvents + [k k+1]];
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

save testTraces_mn_hist outTraces
