% This is just an example, how to define a more complex processing scheme.
% It relates to the 'super speller' example in the WIKI documentation.

bbci= struct;

%% Acquisition: source
% EEG
fs_eeg= 100;
bbci.source.acquire_fcn= @bbci_acquire_bv;
bbci.source.acquire_param= {struct('fs',fs_eeg)};
% NIRS
bbci.source(2).acquire_fcn= @bbci_acquire_nirs;
bbci.source(2).acquire_param= {};
% slow NIRS sampling should not block fast EEG sampling:
bbci.source(2).min_blocklength= 0;

%% Processing of continuous signals: signal
% alpha band-power from EEG
[filt_alpha_b, filt_alpha_a]= butter(5, [8 12]/fs_eeg*2);
bbci.signal.source= 1;
bbci.signal.proc= {{@online_filt, filt_alpha_b, filt_alpha_a}};
% raw EEG signals for ERPs
bbci.signal(2).source= 1;
% theta, alpha, beta band-power from EEG
[filt_tab_b, filt_tab_a]= butters(5, [5 7; 8 11; 18 26]/fs_eeg*2);
bbci.signal(3).source= 1;
bbci.signal(3).proc= {{@online_filterbank, filt_b, filt_a}};
% Oxy from NIRS
bbci.signal(4).source= 2;
bbci.signal(4).proc= {{@proc_extractOxy}};
% Deoxy from NIRS
bbci.signal(5).source= 2;
bbci.signal(5).proc= {{@proc_extractDeoxy}};

%% Feature extraction - feature
% ERD of alpha
bbci.feature.signal= 1;
bbci.feature.proc= {@proc_variance, @proc_logarithm};
bbci.feature.ival= [100 1000];
% ERP for stimulus cues
bbci.feature(2).signal= 2;
bbci.feature(2).proc= {{@proc_baseline, [-200 0]}, ...
                       {@proc_jumpingMeans, [100 130; 130 200; 200 500]}};
bbci.feature(2).ival= [-200 500];
% ERP for feedback response
bbci.feature(2).signal= 2;
bbci.feature(2).proc= {{@proc_baseline, [-200 0]}, ...
                       {@proc_jumpingMeans, [100 130; 130 200; 200 400]}};
bbci.feature(2).ival= [-200 400];
% ERD of alpha, beta, theta
bbci.feature(3).signal= 3;
bbci.feature(3).proc= {@proc_variance, @proc_logarithm};
bbci.feature(3).ival= [-10000 0];
% Bold of Oxy
bbci.feature(4).signal= 4;
bbci.feature(4).proc= {@proc_bold};
bbci.feature(4).ival= [-10000 0];
% Bold of Oxy
bbci.feature(5).signal= 5;
bbci.feature(5).proc= {@proc_bold};
bbci.feature(5).ival= [-10000 0];

%% Classification - classifier
% Attention modulated responds to cues
bbci.classifier.feature= [1 2];
bbci.classifier.C= 'Some trained LDA classifier';
% Detection of error potential
bbci.classifier(2).feature= [3];
bbci.classifier(2).C= 'Some trained LDA classifier';
% quantifying vigilance
bbci.classifier(3).feature= [4 5];
bbci.classifier(3).C= 'Some trained LDA classifier';

%% Transforming classifier output into a control signal - control
% Symbol selection
bbci.control.classifier= 1;
bbci.control.fcn= @bbci_control_ERP_Speller;
bbci.control.param= {struct('nClasses',6, 'nSequences',10)};
bbci.control.condition.marker= [11:16,21:26,31:36,41:46];
% Error detector
bbci.control(2).classifier= 2;
bbci.control(2).fcn= @bbci_control_ErrP_Detector;
bbci.control(2).condition.marker= [100:160];
% Vigilance monitor
bbci.control(3).classifier= 3;

bbci.quit_condition.marker= 255;
