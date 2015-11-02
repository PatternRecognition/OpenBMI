clear all;
OpenBMI % Edit the variable BMI if necessary
global BMI; % check BMI directories

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_RAW_DIR, '\calibration_motorimageryVPkg');
[eeg, eeg.mrk_orig, eeg.hdr]=Load_EEG_data(file,'device','brainVision','fs', 100);
mrk_define={'1','left','2','right','3','foot'};
eeg.mrk=mrk_redefine_class(eeg.mrk_orig, mrk_define); 
eeg.mrk=mrk_select_class(eeg.mrk,{'right', 'left'});


%% CROSS-VALIDATION MODULE
CV.prep={ % commoly applied to training and test data
    'prep_filter', {'frequency', [7 13]} %be applied to all data (before split)
    'prep_segmentation', {'interval', [750 3750]}
    };
CV.train={
    'func_csp', {'nPatterns','3','cov','normal'}
    'func_featureExtraction', {'feature', 'logvar'}
    'classifier_trainClassifier', {'LDA'}
    };
CV.test={
    'func_projection',{}
    'func_featureExtraction',{'feature', 'logvar'}
    'classifier_applyClassifier',{}
    };
% CV.perform={
%     'loss=cal_loss(out, label)'
%     }
CV.option={
'KFold','10'
};

[loss]=eval_crossValidation(eeg, CV); % input : eeg, or eeg_epo


