subject= 'Gabriel';
date_str= '03_07_16';

sub_dir= [subject '_' date_str '/'];
fig_dir= ['preliminary/' sub_dir];

exp_name= 'selfpaced3s_fs';
file= [sub_dir exp_name subject];

global NO_CLOCK
NO_CLOCK= 1;


model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model_RLDA.param= [0 0.001 0.01 0.1];


method_dtct= struct('ilen', 1270);
method_dtct.chans= {'FC3-4', 'CFC5-6', 'C5-6', 'CCP5-6', 'CP5-6','P3-4'};
method_dtct.proc= ['fv= proc_filtBruteFFT(epo, [0.8 3], 128, 150); ' ...
                   'fv= proc_jumpingMeans(fv, 5);'];
method_dtct.jit= [-50, -150];
method_dtct.jit_noevent= [-1200, -1400];
method_dtct.xTrials= [10 10];
method_dtct.separateActionClasses= 0;
method_dtct.combinerFcn= inline('-x');
method_dtct.model= 'linearPerceptron';
%method_dtct.model= model_RLDA;
%method_dtct.msTrials= [3 10 -1];



dsply= struct('E', -1500:10:1000);
dsply.facealpha= 0;
dsply.tubePercent= [10 20 30];

val= struct('train_file', file);
val.test_file= {};
val.train_idx= 0.75;
val.test_idx= [];
val.xTrials= [];

figure(1); clf;
[testTube, outTraces, labels]= ...
    plot_tube_detection(val, method_dtct, dsply);

figure(2); clf;
[testTube, outTraces, labels]= ...
    plot_tube_detection_old(val, method_dtct, dsply);
