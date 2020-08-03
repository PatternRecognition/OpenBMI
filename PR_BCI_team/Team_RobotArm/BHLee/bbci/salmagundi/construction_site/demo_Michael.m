global michaels_w 
file = 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'F#', 'FC#', 'C#', 'CP#', 'P#');

epo= makeEpochs(cnt, mrk, [-1300 0]-120);
no_moto= makeEpochs(cnt, mrk, [-1500 -1000]);
everything = makeEpochs(cnt,mrk,[-1500 -200]);

epo = proc_filtBruteFFT(epo,[0.8,3],128,200);
%no_moto = proc_filtBruteFFT(no_moto,[0.8,3],128);

[epow, michaels_w] = proc_spatialprojection(epo,1,no_moto,'id',2,everything,'smooth');


method.chans= {'F#', 'FC#', 'C#', 'CP#', 'P#'};
method.ival= [-1270 -100];
%method.proc= ['global michaels_w; ' ...
%              'fv= proc_linearDerivation(epo, michaels_w); ' ...
%	      'fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200); ' ...
%              'fv= proc_jumpingMeans(fv, 5);'];
method.proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 200); ' ...
              'fv= proc_jumpingMeans(fv, 5);'];
method.jit= [-50, -100, -150];
method.model= 'LSR';

val.train_file= file;
val.test_file= {};
val.train_idx= [];
val.test_idx= [];
val.xTrials= [1 1];  %% leave-one-out

dsply.E= -1000:50:500;
dsply.facealpha= 0;
clf;
plot_tube_descrimination(val, method, dsply);
