clear all; close all; clc;
OpenBMI('E:\Test_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['E:\Test_OpenBMI\BMI_data\RawEEG'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
SMT=prep_segmentation(CNT, {'interval',[-2000 5000]});

%% Example code for visualization
%
%   ERSP:
%   ersp = visual_ERSP(CNT, {'Channel' , {'C4'}; 'Interval' ,[-2000 5000]});
%   ersp = visual_ERSP(SMT, {'Channel' , {'C4'}});
%
%   Visual_spectrum: 
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Channel'; 'Band' ,[8 10]});
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Frequency'; 'Channel' ,{'C4'}});




