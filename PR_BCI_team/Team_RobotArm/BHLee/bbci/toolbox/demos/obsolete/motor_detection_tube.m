file= 'Gabriel_00_09_05/selfpaced2sGabriel';

method.chans= {'FC5-6', 'C5-6', 'CP5-6', 'P3-4'};
method.ival= [-1270 0];
method.proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 300); ' ...
              'fv= proc_jumpingMeans(fv, 5);'];
method.jit= [-50, -100, -150];
method.jit_noevent= [-700, -750, -800];
method.model= 'LSR';
method.separateActionClasses= 1;
method.combinerFcn= inline('max(x(1,2,:))-x(3,:)');

val.train_file= file;
val.test_file= {};     %% means same file
val.train_idx= 0.75;   %% train on first 75%, test on last 25%
val.test_idx= [];      %% means the rest
val.xTrials= [];       %% no x-validation

dsply.E= -1500:10:500;
dsply.facealpha= 0;

plot_tube_detection(val, method, dsply);
