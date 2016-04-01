clear all;
OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\DemoData'];
%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 100});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan',};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);


CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
EEG.data=prep_filter(EEG.data, {'frequency', [7 13]});
EPO=prep_segmentation(EEG.data, EEG.marker, {'interval', [750 3500]});

prep_addTrials(EPO.x, EPO.m)


%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[EPO_CSP, CSP_W, CSP_D]=func_csp(EPO,{'nPatterns', 3});
EPO_FV=func_featureExtraction(EPO_CSP, 'logvar');

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(EPO_FV,'LDA');

%% TEST DATA LOAD
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
[EEGfb.data, EEGfb.marker, EEGfb.info]=Load_EEG(file,{'device','brainVision';'fs', 100});

[EEGfb.marker, EEGfb.markerOrigin]=prep_defineClass(EEGfb.marker,{'1','left';'2','right';'3','foot';'4','rest'}); 
EEGfb.marker=prep_selectClass(EEGfb.marker,{'right', 'left'});

EEGfb.data=prep_filter(EEGfb.data, {'frequency', [7 13]});
EEGfb.data=func_projection(EEGfb.data, CSP_W);

EPOfb=prep_segmentation(EEGfb.data, EEGfb.marker, {'interval', [750 3500]});
EPOfb_FV=func_featureExtraction(EPOfb, 'logvar');

[cf_out]=func_predict(EPOfb_FV, CF_PARAM);

[loss out]=eval_calLoss(EPOfb_FV.y, cf_out);