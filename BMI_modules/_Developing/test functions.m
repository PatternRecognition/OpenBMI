clear all;
OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\git_OpenBMI\DemoData'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});


%% PRE-PROCESSING MODULE
CNT=prep_filter(CNT, {'frequency', [7 13]});
SMT=prep_segmentation(CNT, {'interval', [-750 3500]});


aCNT=prep_addTrials(CNT, CNT);

aveCNT=prep_average(SMT);

prep_rejectArtifactMAxMin(SMT, 70);