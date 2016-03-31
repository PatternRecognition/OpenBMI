clear all;
OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\OpenBMI_Ver3') % Edit the variable BMI if necessary
global BMI;
%% DATA LOAD MODULE
file=fullfile(BMI.EEG_RAW_DIR, '\smkim0002');
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'fs', 250});

[EEG.marker, EEG.markerOrigin]=prep_defineClass(EEG.marker,{'1','left';'2','right';'3','foot';'4','rest'}); 
EEG.marker=prep_selectClass(EEG.marker,{'right', 'left'});

%% PRE-PROCESSING MODULE
EEG.data=prep_filter(EEG.data, {'frequency', [7 13]});
EPO=prep_segmentation(EEG.data, EEG.marker, {'interval', [750 3500]});