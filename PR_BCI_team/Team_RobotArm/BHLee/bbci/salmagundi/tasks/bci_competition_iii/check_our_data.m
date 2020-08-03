train_file= 'Guido_04_11_01/imag_lettGuido';
test_file= 'Guido_04_11_01/imag_lettfastGuido';

[cnt, mrk, mnt, N]= concatProcessedEEG({train_file, test_file});
N= N/3*2;
mrk= mrk_selectClasses(mrk, 'left','foot');
idxTr= 1:N(1);
idxTe= N(1)+[1:N(2)];
divTr= {{idxTr}};
divTe= {{idxTe}};


%% visual EP

epo= makeEpochs(cnt, mrk, [100 350]);
epo= proc_selectChannels(epo, 'P3,5,7','PCP5,7','PPO7','PO7');
epo= proc_jumpingMeans(epo, 2);

opt= struct('out_trainloss',1, 'outer_ms',1);
model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model_RLDA.param= [0 0.001 0.01 0.1 0.5];

xvalidation(epo, model_RLDA, opt);
xvalidation(epo, model_RLDA, opt, 'divTr',divTr, 'divTe',divTe);


%% CSP approach

csp.clab = {'not','E*','Fp*','FAF*','I*','AF*'};
csp.ival= [750 1750];
csp.band= [7 30];
csp.filtOrder= 5;
csp.nPat= 3;

[b,a]= getButterFixedOrder(csp.band, cnt.fs, csp.filtOrder);
cnt_flt= proc_filt(cnt, b, a);

fv= makeEpochs(cnt_flt, mrk, csp.ival);
fv= proc_selectChannels(fv, csp.clab);
opt_csp= struct('xTrials', [1 5]);
opt_csp.proc= ['fv= proc_csp(fv, ' int2str(csp.nPat) '); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];

xvalidation(proc_selectEpochs(fv, idxTr), 'LDA', opt_csp);
xvalidation(proc_selectEpochs(fv, idxTe), 'LDA', opt_csp);
xvalidation(fv, 'LDA', opt_csp);
xvalidation(fv, 'LDA', opt_csp, 'divTr',divTr, 'divTe',divTe);




test_file= 'Guido_04_11_01/imag_auditoryGuido';

[cnt, mrk, mnt]= loadProcessedEEG(test_file);
mnt= setDisplayMontage(mnt, 'medium')

epo= makeEpochs(cnt, mrk, [-200 2000]);
epo= proc_baseline(epo, [-200 0]);
epo= proc_movingAverage(epo, 100, 'centered');
grid_plot(epo, mnt);

epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);
