clear all;
OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\DemoData'];

aaaaaaaaaaaaaaaaaaaaaaaa;
%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 100});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan',};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
CNT=prep_filter(CNT, {'frequency', [7 13]});
EPO=prep_segmentation(CNT, {'interval', [750 3500]});


%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[EPO_CSP, CSP_W, CSP_D]=func_csp(EPO,{'nPatterns', 3});
EPO_FV=func_featureExtraction(EPO_CSP, 'logvar');

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(EPO_FV,'LDA');

%% TEST DATA LOAD
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEGfb.data, EEGfb.marker, EEGfb.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', 100});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan',};
CNTfb=opt_eegStruct({EEGfb.data, EEGfb.marker, EEGfb.info}, field);

CNTfb=prep_selectClass(CNTfb,{'class',{'right', 'left'}});

CNTfb=prep_filter(CNTfb, {'frequency', [7 13]});
EPOfb=prep_segmentation(CNTfb, {'interval', [750 3500]});

EPOfb=func_projection(EPOfb, CSP_W);
EPOfb=func_featureExtraction(EPOfb, 'logvar');
[cf_out]=func_predict(EPOfb, CF_PARAM);

[loss out]=eval_calLoss(EPOfb.y_dec, cf_out);


