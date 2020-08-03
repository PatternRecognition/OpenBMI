file= 'Gabriel_00_10_04/selfpacedmultiGabriel';

clasIval= [-1000 -150];
filtIval= [-1000 0];                     %% for filtering in epochs

fprintf('loading and filtering will take some time\n');
cnt= readGenericEEG(file);
cnt= proc_filtBackForth(cnt, 'infradelta');
mrk= readMarkerTable(file);
classDef= {[65 70],[3 4]; 'left', 'right'};
mrk= makeClassMarkersMulti(mrk, classDef, 200);

epo_acausal= makeSegments(cnt, mrk, clasIval);

fv= proc_selectChannels(epo_acausal, 'FC#', 'C#', 'CP#');
fv= proc_jumpingMeans(fv, 5, 7);
fprintf(['[acausal filtering of continuous signal, ' ...
         'classification in [%d %d]]\n'], clasIval);
fprintf('3 classes: ');
doXvalidation(fv, 'LDA', [5 10]);

fv.y= fv.y(1:2,:);
fprintf('2 classes: ');
doXvalidation(fv, 'LDA', [5 10]);


[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, filtIval);

fprintf('filtering will take some time\n');
epo_causal= proc_filtBackForth(epo, 'infradelta');

fv= proc_selectChannels(epo_causal, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, clasIval);
fv= proc_jumpingMeans(fv, 5, 7);
fprintf(['[acausal filtering in segments [%d %d], ' ...
         'classification in [%d %d]]\n'], filtIval, clasIval);
fprintf('3 classes: ');
doXvalidation(fv, 'LDA', [5 10]);

fv.y= fv.y(1:2,:);
fprintf('2 classes: ');
doXvalidation(fv, 'LDA', [5 10]);



iv= getIvalIndices([-1000 -150], epo);
evt= 0;

%% for repetition:
evt= evt+1;
subplot(211);
plot(epo_acausal.t(iv), epo_acausal.x(iv,14,evt));
axis tight;
title('with acausal low-pass filtering');
subplot(212);
plot(epo_causal.t(iv), epo_causal.x(iv,14,evt));
axis tight;
hold on;
plot(epo.t(iv), epo.x(iv,14,evt), 'r');
hold off;
title('with causal low-pass filtering');
