clear all;
OpenBMI % Edit the variable BMI if necessary
global BMI;
%% DATA LOAD MODULE
file=fullfile(BMI.EEG_RAW_DIR, '\calibration_motorimageryVPkg');
[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.mrk=mrk_select_class(eeg.mrk,{'right', 'left'});

%% PRE-PROCESSING MODULE
eeg_flt=prep_filter(eeg, 'frequency', [7 13]);
eeg_epo=prep_segmentation(eeg_flt, 'interval', [750 3500]);


%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[eeg_csp, CSP_W, CSP_D]=func_csp(eeg_epo,'nPatterns', '3', 'policy', 'normal');
eeg_ft=func_featureExtraction(eeg_csp, 'logvar');

%% CLASSIFIER MODULE
[CF_PARAM]=classifier_trainClassifier(eeg_ft,'LDA');

%% TEST DATA LOAD
file='feedback_motorimageryVPkg';
[eegfb, eegfb.mrk_orig, eegfb.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eegfb.mrk=mrk_redefine_class(eegfb.mrk_orig,mrk_define); 
eegfb.mrk=mrk_select_class(eegfb.mrk, {'right','left'});

eegfb=prep_filter(eegfb, 'frequency', [7 13]);
eegfb=func_projection(eegfb, CSP_W);

eegfb=prep_segmentation(eegfb, 'interval', [750 3500]);
eegfb=func_featureExtraction(eegfb, 'logvar');
[cf_out]=classifier_applyClassifier(eegfb, CF_PARAM);
[loss out]=eval_calLoss(eegfb.y, cf_out);

    




