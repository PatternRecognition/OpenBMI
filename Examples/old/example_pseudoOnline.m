clear all; clc; close all;

OpenBMI % Edit the variable BMI if necessary
global BMI;
%% TRAIN DATA LOAD
file=fullfile(BMI.EEG_RAW_DIR, '\test1');
[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.mrk=mrk_select_class(eeg.mrk, {'right','left'});

%% FEEDBACK DATA LOAD
file='test2';
[eegfb, eegfb.mrk_orig, eegfb.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eegfb.mrk=mrk_redefine_class(eegfb.mrk_orig,mrk_define); 
eegfb.mrk=mrk_select_class(eegfb.mrk, {'right','left'});
eegfb=prep_filter(eegfb, 'frequency', [7 13]);

%% OPTIONS
online.train={
    'prep_filter', {'frequency', [7 13]}
    'prep_segmentation', {'interval', [750 3750]}
    'func_csp', {'nPatterns','3','policy', 'normal'}
    'func_featureExtraction', {'feature', 'logvar'}
    'classifier_trainClassifier', {'LDA'}
    };
online.apply={
    'func_projection',{}  
    'func_featureExtraction',{'feature', 'logvar'}
    'classifier_applyClassifier',{}
    };

online.option={
'windowSize', '100' % 1s
'paradigm', 'MotorImagery'
'Feedback','off'
};

[cf_out]=MotorImagery_pseudoOnline(eeg, eegfb, online);
