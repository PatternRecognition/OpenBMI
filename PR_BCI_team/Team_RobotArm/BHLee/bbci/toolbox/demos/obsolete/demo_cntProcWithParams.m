This is MURKS (so far) - do NOT read on:

file= 'Matthias_03_10_22/imagMatthias';

csp.clab = {'not','E*','Fp*','FAF*','I*','AF*'};
csp.ival= [750 3500];
csp.filtOrder= 5;
csp.nPat= 3;

[cnt,mrk]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, csp.clab);
cntmrk= mergeCntMrk(cnt, mrk);

proc.eval= ['[b,a]= butter(' int2str(csp.filtOrder) ', band/' ...
             int2str(fv.fs) '*2); ' ...
            'fv= proc_filt(fv, b, a); ' ...
            'fv= makeEpochs(fv, fv, ' int2str(csp.ival) '); '];
proc.train= ['[fv,csp_w]= proc_csp(fv, nPat); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv,csp_w); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
proc.memo= {'csp_w'};
proc.param(1).var= 'band';
proc.param(1).value= {[7 13], [8 13], [9 13]};


opt= struct('xTrials',[1 5], 'verbosity',2);


%% fake
best_proc= select_proc(fv, 'LDA', {proc1, proc2}, opt);
xvalidation(fv, 'LDA', opt, 'proc',best_proc);

%% less fake
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2}, 'outer_ms',1);

%% no fake - take a break
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2});
