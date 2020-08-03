file= 'Gabriel_01_12_12/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
Mrk= readMarkerTable(file);

lastSeg= Mrk.pos(max(find(Mrk.toe==252)));
train_idx= find(mrk.pos<lastSeg);
test_idx= find(mrk.pos>lastSeg);

epo= makeSegments(cnt, mrk, [-1280 0]-120);
fv= proc_selectChannels(epo, 'FC5-6','C5-6','CCP5-6','CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_jumpingMeans(fv, 5);

fv.divTr= {{train_idx}};
fv.divTe= {{test_idx}};

doXvalidationPlus(fv, 'LSR', []);



return

shift= 0;
nJits= 8;
step= -30;
%nJits= 4;
%step= -80;
nBase= length(mrk.toe);
epo= makeSegments(cnt, mrk, [-1280 0]+shift, step*(1:nJits-1));
fv= proc_selectChannels(epo, 'FC5-6','C5-6','CCP5-6','CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 5], 128, 150);
fv= proc_jumpingMeans(fv, 5);

fv.divTr= {{jitteredIndices(train_idx, nBase*(1:nJits-1))}};
fv.divTe= {{jitteredIndices(test_idx, nBase*(1:nJits-1))}};

doXvalidationPlus(fv, 'LSR', []);
