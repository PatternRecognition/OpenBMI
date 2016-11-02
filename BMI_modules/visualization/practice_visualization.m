%% Example code for visualization function
% Desciption:
%   This is example code for using visualization function. There are include
%   from data load to pre-processing stop (The basic brain signal processing step)
%
% See also:
%    opt_cellToStruct , opt_eegStruct, visual_ERSP, visual_spectrum, visual_scalpPlot
% Reference:
%       M. -H. Lee, S. Fazli, K.-T. Kim, and S.-W. Lee, ¡°Development of 
%       an open source platform for brain-machine interface: OpenBMI,¡± 
%       Proc. 4th IEEE International Winter Conference on Brain-Computer Interface, 
%       Korea, February, 2016, pp. 1-2
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
%
%%
% LOAD THE OpenBMI TOOLBOX
clear all; close all; clc;
OpenBMI('E:\Test_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['E:\Test_OpenBMI\BMI_data\RawEEG'];

% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

% if you can redefine the marker information after Load_EEG function
% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT = prep_filter(CNT, {'frequency', [8 13];'fs',100})
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
% CNT = prep_selectChannels(CNT, {'Index',[1 :64]});
SMT=prep_segmentation(CNT, {'interval',[-3000 5000]});

%% visualization code
%
%   ERSP:
%   ersp = visual_ERSP(CNT, {'Channel' , {'C3'}; 'Interval' ,[-2000 5000]});
%   ersp = visual_ERSP(SMT, {'Channel' , {'C4'}});
%
%   Power spectrum: 
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Channel'; 'Band' ,[8 10]});
%   visuspect = visual_spectrum(SMT , {'Xaxis' , 'Time'; 'Yaxis' , 'Frequency'; 'Channel' ,{'C4'}});

%  ScalpPlot:
%     visual_scalpPlot(SMT,CNT, {'Ival' , [-2000 : 1000: 4000]});
%     

