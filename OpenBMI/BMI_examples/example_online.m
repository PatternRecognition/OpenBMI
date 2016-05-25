clear all;
OpenBMI % Edit the variable BMI if necessary
global BMI;
%% DATA LOAD MODULE
file=fullfile(BMI.EEG_RAW_DIR, '\calibration_motorimageryVPkg');
[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.mrk=mrk_select_class(eeg.mrk,{'right', 'left'});


%% CROSS-VALIDATION MODULE
online.train={
    'prep_filter', {'frequency', [10 14]}
    'prep_segmentation', {'interval', [2000 4000]}
    'func_csp', {'nPatterns','3','policy', 'normal'}
    'func_featureExtraction', {'feature', 'logvar'}
    'classifier_trainClassifier', {'LDA'}
    };
online.apply={
    'prep_filter', {'frequency', [10 14]}
    'func_projection',{}
    'func_featureExtraction',{'feature', 'logvar'}
    'classifier_applyClassifier',{}
    };
online.option={
'device', 'BrainVision'
'paradigm', 'MotorImagery'
'Feedback','off'
'host', 'WIN-9UGFJQRFCNV' %important
'port','51244'
};

MotorImagery_online(eeg, online);
