

fvc= proc_appendFeatures(fv, fv2);

doXvalidationPlus(fvc, {'catCombiner', 'LDA'}, xTrials, 3);

model= struct('classy',{{'catCombiner', 'RLDA'}}, 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fvc, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fvc, classy, xTrials, 3);


doXvalidationPlus(fvc, {'probCombiner', 'LDA'}, xTrials, 3);



model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
