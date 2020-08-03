file= 'Thorsten_02_07_31/imagmultimodalThorsten';
classes= {'leg','auditory'};
ar_ival= [500 3000];
ar_order= 5;

[cnt,mrk]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not', 'E*');
clInd= find(ismember(mrk.className, classes));
cli= find(any(mrk.y(clInd,:)));
mrk_cl= pickEvents(mrk, cli);

cnt_flt= proc_laplace(cnt, 'small', ' lap', 'filter all');
cnt_flt= proc_selectChannels(cnt_flt, ...
                             'FC3-4','C3-4','CP3-4','P3-4','PO#','O#');
[b,a]= getButterFixedOrder(ar_band, cnt_flt.fs, 6);
cnt_flt= proc_filt(cnt_flt, b, a);
fv= makeSegments(cnt_flt, mrk_cl, ar_ival);
clear cnt_flt
fv= proc_rcCoefsPlusVar(fv, ar_order);

model.classy= 'RLDA';
model.param= [0 0.00001 0.001];
model.msDepth= 1;
classy= selectModel(fv, model, [3 10 round(9/10*size(fv.y,2))]);

