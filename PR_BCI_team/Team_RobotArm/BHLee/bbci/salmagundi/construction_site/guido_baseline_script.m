[cnt,mrk] = loadProcessedEEG(file);

cnt = proc_selectChannels(cnt,'not','E*','O*','I*','T*','Fp*','AF*','PP*','FA*','PO*');

epo = makeEpochs(cnt, mrk, [-1300 0]-120);
epo.equi.idcs = getEventPairs(mrk, 5000);

fv1 = proc_filtBruteFFT(epo, [0.8 3], 128, 150);
fv1 = proc_jumpingMeans(fv1, 5);

fv2 = proc_filtBruteFFT(epo, [0.8 3], 128, 100);
fv2 = proc_meanAcrossTime(fv2);

fv3 = proc_selectIval(epo, [-600 -400]);
fv3 = proc_meanAcrossTime(fv3);

fv4= proc_catFeatures(fv2, fv3);

fv5= proc_catFeatures(fv1, fv3);

model.classy = 'RLDA';
model.param.index = 2;
model.param.value = [0:0.2:1].^3;
model.msDepth = 5;

fprintf('\r0/5');

% setting 1
classy = selectModel(fv1, model, [3 10 -1], 0);
[te1,st1] = doXvalidationPlus(fv1, classy, [10 10]);
fprintf('\r1/5');


% setting 2
classy = selectModel(fv2, model, [3 10 -1], 0);
[te2,st2] = doXvalidationPlus(fv2, classy, [10 10]);
fprintf('\r2/5');

% setting 3
classy = selectModel(fv3, model, [3 10 -1], 0);
[te3,st3] = doXvalidationPlus(fv3, classy, [10 10]);
fprintf('\r3/5');

% setting 4
classy = selectModel(fv4, model, [3 10 -1], 0);
[te4,st4] = doXvalidationPlus(fv4, classy, [10 10]);
fprintf('\r4/5');


% setting 5
classy = selectModel(fv5, model, [3 10 -1], 0);
[te5,st5] = doXvalidationPlus(fv5, classy, [10 10]);
fprintf('\r');

fprintf('Common setting: \t%2.1f\n',te1(1));
fprintf('Mean about Time: \t%2.1f\n',te2(1));
fprintf('Mean earlier: \t\t%2.1f\n',te3(1));
fprintf('Combine 2 and 3: \t%2.1f\n',te4(1));
fprintf('Combine 1 and 3: \t%2.1f\n',te5(1));

