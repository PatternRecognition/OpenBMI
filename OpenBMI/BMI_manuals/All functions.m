%% OPTION FUNCTION

%% DATA LOAD
startup_openbmi % Edit the variable BMI if necessary
[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.mrk=mrk_select_class(eeg.mrk,{'right', 'left'});

%% PRE-PROCESSING MODULE
eeg=proc_filter(eeg, 'frequency', [10 14]);
eeg_epo=proc_epoching(eeg, 'interval', [2000 4000]);
eeg_epo=proc_select_channel(eeg_epo,[1:24]);
eeg_epo=proc_delect_channel(eeg_epo,[7 8 10]);

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[eeg_csp, CSP_W, CSP_D]=proc_csp(eeg_epo,'nPatterns','3');
dat=proc_projection(eeg_epo, CSP_W);
eeg_feature=proc_feature_extraction(eeg_csp, 'feature', 'logvar');

%% CLASSIFIER MODULE
[CF_PARAM]=proc_train_classifier(eeg_feature,'LDA');