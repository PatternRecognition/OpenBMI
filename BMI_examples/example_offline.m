clear all;
OpenBMI('D:\OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['D:\OpenBMI\Data\MI'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function 
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
CNT=prep_filter(CNT, {'frequency', [7 13]});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% VISUALIZATION MODULE
figure(1); visual_TopoPlot(SMT, CNT.x(2500,:), CNT);

startup_bbci_toolbox; % for comparision
mnt = mnt_setElectrodePositions(CNT.chan);
figure(2); plot_scalp(mnt, CNT.x(2500,:)); 

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

%% TEST DATA LOAD
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEGfb.data, EEGfb.marker, EEGfb.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan',};
CNTfb=opt_eegStruct({EEGfb.data, EEGfb.marker, EEGfb.info}, field);

CNTfb=prep_selectClass(CNTfb,{'class',{'right', 'left'}});

CNTfb=prep_filter(CNTfb, {'frequency', [7 13]});
SMTfb=prep_segmentation(CNTfb, {'interval', [750 3500]});

SMTfb=func_projection(SMTfb, CSP_W);
FTfb=func_featureExtraction(SMTfb, {'feature','logvar'});
[cf_out]=func_predict(FTfb, CF_PARAM);

[loss out]=eval_calLoss(FTfb.y_dec, cf_out);


