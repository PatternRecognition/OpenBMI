% !! can be replaced by demo_bci_apply_SMR_Lap_C34_bp2 !!

% Subject-independent classifier
cfy_file= [EEG_RAW_DIR 'subject_independent_classifiers/season10/Lap_C34_bp2_LR'];
% EEG file used of offline simulation of online processing
eeg_file= 'VPkg_08_08_07/imag_arrowVPkg';
[cnt, mrk_orig]= eegfile_loadMatlab(eeg_file, 'vars',{'cnt','mrk_orig'});

% For demonstration purpose, we do not use a function to convert the old
% classifier to the new format.
% Instead we get some parameters from the classifier file and build up
% the new bbci structure manually.
S= load(cfy_file);
clab= S.cont_proc.clab;
A= S.cont_proc.procParam{1}{1};
filt_b= S.cont_proc.procParam{2}{1};
filt_a= S.cont_proc.procParam{2}{2};
C= S.cls.C;


bbci= struct;
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk_orig};

bbci.signal.clab= clab;
bbci.signal.proc= {{@online_linearDerivation, A},
                   {@online_filterbank, filt_b, filt_a}};

bbci.feature.proc= {@proc_variance, @proc_logarithm};
bbci.feature.ival= [-500 0];

bbci.classifier.C= C;

bbci_apply(bbci);
