file = 'Gabriel_01_10_15/imagGabriel';
%file = 'Soeren_02_08_13/imagmultiSoeren';

[cnt, mrk] = loadProcessedEEG(file);

[b,a]= getButterFixedOrder([7 20], cnt.fs);
cnt = proc_filt(cnt, b, a);

epo = makeSegments(cnt, mrk, [1000 3000]);

model.classy = {'csp',4,size(epo.x,1),'LDA',0};
model.param.index = 6;
model.param.scale = 'lin';
model.param.value = 0:0.2:1;
model.msDepth = 3;

model = selectModel(epo,model,[3 10 round(size(epo.y,2)*0.9)]);

doXvalidationPlus(epo,model,[10 10]);



file = 'Soeren_02_08_13/imagmultiSoeren';

[cnt, mrk] = loadProcessedEEG(file);

[b,a]= getButterFixedOrder([8 13], cnt.fs);
cnt = proc_filt(cnt, b, a);

clInd= [1 2];

cli= find(any(mrk.y(clInd,:)));
mrk_cl= pickEvents(mrk, cli);
fprintf('<%s> vs <%s>\n', mrk_cl.className{:});

epo = makeSegments(cnt, mrk_cl, [1000 2000]);

model.classy = {'csp', 2, size(epo.x,1), 'LDA', 0};
model.param.index = 6;
model.param.scale = 'lin';
model.param.value = 0:0.2:1;
model.msDepth = 1;

classy = selectModel(epo, model, [3 10 round(size(epo.y,2)*0.9)]);
doXvalidationPlus(epo, classy, [3 10]);
