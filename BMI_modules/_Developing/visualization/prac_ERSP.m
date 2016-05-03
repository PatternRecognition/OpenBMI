clear all; close all; clc;
% OpenBMI('E:\Test_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
% global BMI;
% BMI.EEG_DIR=['E:\Test_OpenBMI\BMI_data\RawEEG'];
% 
% %% DATA LOAD MODULE
% file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
% marker={'1','left';'2','right';'3','foot';'4','rest'};
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});
% 
% %% if you can redefine the marker information after Load_EEG function
% %% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)
% 
% field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
% CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
% SMT=prep_segmentation(CNT, {'interval',[0 4000]});
% a = plotERSP(SMT , {'freqBinWidth' , 2;'spectralSize' , 200; 'spectralStep' , 50; }) ;
%% Test bci2000 data
load('E:\Test_OpenBMI\visualization\x1.mat');
load('E:\Test_OpenBMI\visualization\x2.mat');
A.x = x1;
A.fs = 160;
A.y_dec=1;
A.x = permute(A.x, [1, 3, 2]);

B.x = x2;
B.fs = 160;
B.y_dec=2;
B.x = permute(B.x, [1, 3, 2]);

C.x = cat(2 , A.x,B.x);
C.fs = 160;
C.y = [A.y_dec B.y_dec];
a = plotERSP(C , {'freqBinWidth' , 2;'spectralSize' , 333; 'spectralStep' , 166; }) ;



%% stardard code
% sampleFrequency=CNT.fs;
% modelOrder = 18+round(sampleFrequency/100);
% setting.hpCutoff=-1;
% settings.freqBinWidth=2;
% lp_cutoff=(sampleFrequency/2)-10;%  ≥°¿Ã ¬©∑¡º≠ ¡Ÿ¿”
% settings.trend=1;
% 
% parms = [modelOrder, setting.hpCutoff+settings.freqBinWidth/2, ...
%     lp_cutoff-settings.freqBinWidth/2, settings.freqBinWidth, ...
%     round(settings.freqBinWidth/.2), settings.trend, sampleFrequency];
% 
% 
% 
% memparms = parms;
% 
% 
% [nD nTr nCh]=size(SMT.x)
% spectral_stepping=200;
% spectral_size=200;
% % datalength=round(nD/spectral_stepping)-1
% 
% memparms(8) = spectral_stepping;
% memparms(9) = (spectral_size/spectral_stepping);% µﬁ∫Œ∫– ¬©∑¡º≠ ¡Ÿ¿”
% 
% idx=find(SMT.y_dec==1);
% idx2 = find(SMT.y_dec==2);
% C1=prep_selectTrials(SMT,idx);
% C2=prep_selectTrials(SMT,idx2);
% 
% 
% 
% for i=1:nTr/26
%     dat=C1.x(:,i,:);
%     dat=squeeze(dat);
%     [trialspectrum, freq_bins] = mem( dat, memparms );
%     tm1=mean(trialspectrum, 3);
%     dat_c1(:,:,i)=tm1;
% end
% 
% mean_c1=mean(dat_c1,3);
% xData=freq_bins
% 
% 
% dispmin=min(min(mean_c1));
% dispmax=max(max(mean_c1));
% num_channels=size(mean_c1, 2);
% data2plot=mean_c1;
% data2plot=cat(2, data2plot, zeros(size(data2plot, 1), 1));
% data2plot=cat(1, data2plot, zeros(1, size(data2plot, 2)));
% 
% xData(end+1) = xData(end) + diff(xData(end-1:end));
% surf(xData(4:end), [1:num_channels + 1], data2plot(4:end,:)');
% view(2);
% colormap jet;
% colorbar;
% 
% 
% 
% %% class 2 
% for i=1:nTr/2
%     dat=C2.x(:,i,:);
%     dat=squeeze(dat);
%     [trialspectrum, freq_bins] = mem( dat, memparms );
%     tm1=mean(trialspectrum, 3);
%     dat_c1(:,:,i)=tm1;
% end
% 
% mean_c1=mean(dat_c1,3);
% xData=freq_bins
% 
% 
% dispmin=min(min(mean_c1));
% dispmax=max(max(mean_c1));
% num_channels=size(mean_c1, 2);
% data2plot=mean_c1;
% data2plot=cat(2, data2plot, zeros(size(data2plot, 1), 1));
% data2plot=cat(1, data2plot, zeros(1, size(data2plot, 2)));
% 
% xData(end+1) = xData(end) + diff(xData(end-1:end));
% figure;
% surf(xData(4:end), [1:num_channels + 1], data2plot(4:end,:)');
% view(2);
% colormap jet;
% colorbar;



