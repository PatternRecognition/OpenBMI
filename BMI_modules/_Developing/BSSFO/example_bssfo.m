clear all;
OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function 
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});



%
nFT = 9; % filter bank number
Filters = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40];
[FilterBand]=func_fbcsp(SMT, {'classes', {'right', 'left'}; 'Filters', Filters; 'numCSPPatterns', 2; 'numFeature', nFT; })



%%
[FilterBand]=func_bssfo(SMT, {'classes', {'right', 'left'};'frequency', {[0.1 5],[1 10], [8 15]}; 'std', {1, 5, 10}; ...
    'numBands', 30; 'numCSPPatterns', 2; 'numIteration', 30});

CNT=prep_filter(CNT, {'frequency', FilterBand.sample(:,1)});
SMT=prep_segmentation(CNT, {'interval', [750 3500]});

%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

%% TEST DATA LOAD
file=fullfile(BMI.EEG_DIR, '\feedback_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot'};
[EEGfb.data, EEGfb.marker, EEGfb.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan',};
CNTfb=opt_eegStruct({EEGfb.data, EEGfb.marker, EEGfb.info}, field);

CNTfb=prep_selectClass(CNTfb,{'class',{'right', 'left'}});

CNTfb=prep_filter(CNTfb, {'frequency', FilterBand.sample(:,1)});
SMTfb=prep_segmentation(CNTfb, {'interval', [750 3500]});


SMTfb=func_projection(SMTfb, CSP_W);
FTfb=func_featureExtraction(SMTfb, {'feature','logvar'});
[cf_out]=func_predict(FTfb, CF_PARAM);

[loss out]=eval_calLoss(FTfb.y_dec, cf_out);