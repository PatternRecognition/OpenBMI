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
SMT=prep_segmentation(CNT, {'interval',[0 4000]});
ersp = plotERSP(SMT , {'Xaxis' , 'Frequency'; 'Yaxis' , 'Channel'});


%% Test bci2000 data
% load('E:\Test_OpenBMI\visualization\x1.mat');
% load('E:\Test_OpenBMI\visualization\x2.mat');
% A.x = x1;
% A.fs = 160;
% A.y_dec=1;
% A.x = permute(A.x, [1, 3, 2]);
% 
% B.x = x2;
% B.fs = 160;
% B.y_dec=2;
% B.x = permute(B.x, [1, 3, 2]);
% 
% C.x = cat(2 , A.x,B.x);
% C.fs = 160;
% C.y = [A.y_dec B.y_dec];
% a = plotERSP(C , {'freqBinWidth' , 2;'spectralSize' , 333; 'spectralStep' , 166; }) ;