file= 'Matthias_03_10_22/imagMatthias';

csp.clab = {'not','E*','Fp*','FAF*','I*','AF*'};
csp.ival= [750 3500];
csp.band= [7 30];
csp.filtOrder= 5;

[cnt,mrk]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, csp.clab);
[b,a]= butter(csp.filtOrder, csp.band/cnt.fs*2);
cnt= proc_filt(cnt, b, a);
fv= makeEpochs(cnt, mrk, csp.ival);

proc1= struct;
proc1.train= ['[fv,csp_w]= proc_csp(fv, nPat); ' ...
              'fv= proc_variance(fv); ' ...
              'fv= proc_logarithm(fv);'];
proc1.apply= ['fv= proc_linearDerivation(fv,csp_w); ' ...
              'fv= proc_variance(fv); ' ...
              'fv= proc_logarithm(fv);'];
proc1.memo= {'csp_w'};
proc1.param(1).var= 'nPat';
proc1.param(1).value= {1 2 3 4};

proc2= copy_struct(proc1, 'memo','param');
proc2.train= ['[fv,csp_w]= proc_csp(fv, nPat); ' ...
              'fv= proc_variance(fv); '];
proc2.apply= ['fv= proc_linearDerivation(fv,csp_w); ' ...
              'fv= proc_variance(fv); '];

opt= struct('xTrials',[1 5], 'verbosity',2);


%% fake
best_proc= select_proc(fv, 'LDA', {proc1, proc2}, opt);
xvalidation(fv, 'LDA', opt, 'proc',best_proc);

%% less fake
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2}, 'outer_ms',1);

%% no fake - take a break
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2});
