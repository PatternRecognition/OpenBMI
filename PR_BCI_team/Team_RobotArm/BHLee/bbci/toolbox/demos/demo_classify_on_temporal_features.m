fprintf('Read the code of this script and execute it blockwise.\n');

file= 'Gabriel_00_09_05/selfpaced2sGabriel';
[cnt,mrk,mnt]= loadProcessedEEG(file);

%% The goal with the data set is to discriminate between left and
%% right hand finger taps *before* the actual movement start. Investigation
%% has shwon that (in this experiment) EMG activity starts around 120ms
%% before keypress.
tcl= -120;

epo= makeEpochs(cnt, mrk, [-1280 0]+tcl);

fv= proc_selectChannels(epo, 'FC5-6','C5-6','CP5-6','P3,4');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);
xvalidation(fv, 'LDA');

%% more to come ...
