file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt,mrk]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not','E*');
tcl= -120;

fv= makeEpochs(cnt, mrk, [-990 0]+tcl);
proc1= struct('eval', ['fv= proc_filtBruteFFT(fv, band, 100, len); ' ...
                       'fv= proc_jumpingMeans(fv, 5);']);
proc1.param(1).var= 'band';
proc1.param(1).value= {[0.8 3], [0.8 4], [0.8 5]};
proc1.param(2).var= 'len';
proc1.param(2).value= {100 150 200};

proc2= proc1;
proc2.eval= ['fv= proc_baseline(fv, 150, ''beginning''); ' ...
             'fv= proc_filtBruteFFT(fv, band, 100, len); ' ...
             'fv= proc_jumpingMeans(fv, 5);'];

opt= struct('xTrials',[1 5], 'verbosity',2);

%% fake
best_proc= select_proc(fv, 'LDA', {proc1, proc2}, opt);
xvalidation(fv, 'LDA', opt, 'proc',best_proc);

%% less fake
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2}, 'outer_ms',1);

%% no fake - take a break
xvalidation(fv, 'LDA', opt, 'proc',{proc1, proc2});



model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1];

%% fake
[best_proc, classy]= select_proc(fv, model_RLDA, {proc1, proc2}, opt);
fprintf('selected: '); disp_proc(best_proc);
fprintf('with classifier: %s\n', toString(classy));
xvalidation(fv, classy, opt, 'proc',best_proc);
xvalidation(fv, model_RLDA, opt, 'proc',best_proc);

%% less fake
xvalidation(fv, model_RLDA, opt, 'proc',best_proc, 'outer_ps',1);

%% no cake, but take some cups of coffee
xvalidation(fv, model_RLDA, opt, 'proc',best_proc);
