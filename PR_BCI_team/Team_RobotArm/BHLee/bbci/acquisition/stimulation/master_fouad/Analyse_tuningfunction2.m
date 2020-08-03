% Load Raw EEG Data
clear all;clc;close all;
date='_08_07_04\';
VP_CODE='VPjq';
file = 'fingertipRight';
load(['C:\eeg_data\',VP_CODE,date,'Mat_files\',file,VP_CODE]);

Para.modfreq=[17:2:33];
% Define channels thats used for tuningfunction
plot_chan_lap ={'CCP3 lap','CCP4 lap'};

cnt=dat;
warning('off','all')
saves=1;

%% Define the markers
stimDef = {['S ',int2str(Para.modfreq(1))],['S ',int2str(Para.modfreq(2))],['S ',int2str(Para.modfreq(3))],['S ',int2str(Para.modfreq(4))],['S ',int2str(Para.modfreq(5))],['S ',int2str(Para.modfreq(6))],['S ',int2str(Para.modfreq(7))],['S ',int2str(Para.modfreq(8))],['S ',int2str(Para.modfreq(9))] ... 
    ;['Mod.freq',int2str(Para.modfreq(1))],['Mod.freq',int2str(Para.modfreq(2))],['Mod.freq',int2str(Para.modfreq(3))],['Mod.freq',int2str(Para.modfreq(4))],['Mod.freq',int2str(Para.modfreq(5))],['Mod.freq',int2str(Para.modfreq(6))],['Mod.freq',int2str(Para.modfreq(7))],['Mod.freq',int2str(Para.modfreq(8))],['Mod.freq',int2str(Para.modfreq(9))]};

mrk_stim= mrk_defineClasses(mrk, stimDef);

ival = [0 2000];        % Epoch lenght in ms
fft_band = [6 35];      % fft interval

lap_filter = 'large';   
filter_chan = 'filter all';
channel_list ={'C3-CP3','Cz-CPz','C4-CP4'};

% Reference markers and interval
stimDef_ref = {'S102';'REF'};
mrk_stim_ref = mrk_defineClasses(mrk, stimDef_ref);
ival_ref = [0 5000];

% Artifact rejection
figure('Name','artifacts')
[mrk_clean, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk_stim, ival, 'visualize',1,'whiskerlength', 3);

figure;
[mrk_clean_ref, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk_stim_ref, ival_ref, 'visualize',1,'whiskerlength', 1);

% Epoching the reference trials
epo_ref = cntToEpo(cnt, mrk_clean_ref, ival_ref);

% laplace filter the ref trials 
epo_ref_lap = proc_laplace(epo_ref,lap_filter,' lap',filter_chan);

% Epoching stimuli trials and laplace filter 
epo = cntToEpo(cnt, mrk_clean, ival);
epo_lap = proc_laplace(epo,lap_filter,' lap',filter_chan);

% epo= proc_movingAverage(epo, 10,'centered');

epo_bi= proc_bipolarChannels(epo, channel_list);

% Spectrum of all filteret data
epo_spec= proc_spectrum(epo, fft_band, kaiser(cnt.fs,2));
epo_spec_ref= proc_spectrum(epo_ref, fft_band, kaiser(cnt.fs,2));
epo_lap_spec= proc_spectrum(epo_lap, fft_band, kaiser(cnt.fs,2));
epo_spec_ref_lap = proc_spectrum(epo_ref_lap, fft_band, kaiser(cnt.fs,2));
epo_bi_spec= proc_spectrum(epo_bi, fft_band, kaiser(cnt.fs,2));

% % Calculating r squared value
% epo_lap_spec_r= proc_r_square_signed(proc_selectClasses(epo_lap_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));
% epo_spec_r= proc_r_square_signed(proc_selectClasses(epo_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));
% epo_spec_bi_r= proc_r_square_signed(proc_selectClasses(epo_bi_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));

% [DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp3(epo,'patterns',3);
% DAT_spec= proc_spectrum(DAT, fft_band, kaiser(cnt.fs,2));

% Montages for grid plot
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'tiny');
mnt= mnt_shrinkNonEEGchans(mnt);

mnt1= getElectrodePositions(epo.clab);
mnt1= mnt_setGrid(mnt1, 'large');
mnt1= mnt_shrinkNonEEGchans(mnt1);

mnt_bi= getElectrodePositions(epo_bi.clab);
mnt_bi= mnt_setGrid(mnt_bi, 'tiny');
mnt_bi= mnt_restrictMontage(mnt_bi, epo_bi.clab);

mnt_lap= getElectrodePositions(epo_lap.clab);
mnt_lap= mnt_setGrid(mnt_lap, 'medium');

mnt_lap_1= getElectrodePositions(epo_lap.clab);
mnt_lap_1= mnt_setGrid(mnt_lap_1, 'large');

opt_spec= defopt_spec;
opt_grid= defopt_erps;

% Tuningfunc

epo_lap_avg = proc_average(epo_lap,'classes',epo_lap.className);
epo_ref_lap_avg = proc_average(epo_ref_lap);

% epo_avg = proc_average(epo,'classes',epo.className);
% epo_ref_avg = proc_average(epo_ref);
% 
% epo_avg_lap = proc_laplace(epo_avg,lap_filter,'lap',filter_chan);
% epo_ref_avg_lap = proc_laplace(epo_ref_avg,lap_filter,'lap',filter_chan);
% 
% epo_bi_avg= proc_bipolarChannels(epo_avg, channel_list);
% epo_bi_avg_ref= proc_bipolarChannels(epo_ref_avg, channel_list);
% 
% epo_bi_avg_spec = proc_spectrum(epo_bi_avg,fft_band, kaiser(cnt.fs,2));
% epo_bi_avg_ref_spec = proc_spectrum(epo_bi_avg_ref,fft_band, kaiser(cnt.fs,2));

epo_spec_lap_avg_1 = proc_spectrum(epo_lap_avg,fft_band, kaiser(cnt.fs,2));
epo_spec_ref_lap_avg_1 = proc_spectrum(epo_ref_lap_avg,fft_band, kaiser(cnt.fs,2));

% epo_avg_lap_spec = proc_spectrum(epo_avg_lap,fft_band, kaiser(cnt.fs,2));
% epo_ref_avg_lap_spec = proc_spectrum(epo_ref_avg_lap,fft_band, kaiser(cnt.fs,2));


% figure('Name','Average before spec');grid_plot(epo_bi_avg_spec);
% figure('Name','Average before spec (ref)');grid_plot(epo_bi_avg_ref_spec);

figure('Name','Average before spec');grid_plot(epo_spec_lap_avg_1,mnt_lap);
figure('Name','Average before spec (ref)');grid_plot(epo_spec_ref_lap_avg_1,mnt_lap);

figure('Name','Average before spec (large)');grid_plot(epo_spec_lap_avg_1,mnt_lap_1);
figure('Name','Average before spec (ref) (large)');grid_plot(epo_spec_ref_lap_avg_1,mnt_lap_1);

% figure('Name','Average after spec');grid_plot(epo_lap_spec,mnt_lap);
% figure('Name','Average after spec (ref)');grid_plot(epo_spec_ref_lap,mnt_lap);
% 
% figure('Name','Average before lap');grid_plot(epo_avg_lap_spec,mnt_lap);
% figure('Name','Average before lap (ref)');grid_plot(epo_ref_avg_lap_spec,mnt_lap);

%%


% Find index values
ind_lap = chanind(epo_lap,plot_chan_lap);

% Find max min values 
max_1 = max(max(mean(epo_lap_spec.x,3)));
min_1 = min(min(mean(epo_lap_spec.x,3)));

max_1_ref = max(max(mean(epo_spec_ref_lap.x,3)));
min_1_ref = min(min(mean(epo_spec_ref_lap.x,3)));

max_max = max(max_1,max_1_ref);
min_min = min(min_1,min_1_ref);

max_2 = max(epo_spec_lap_avg_1.x(:,ind_lap(1),:));
min_2 = min(epo_spec_lap_avg_1.x(:,ind_lap(1),:));

max_2_ref = max(epo_spec_ref_lap_avg_1.x(:,ind_lap(1),:));
min_2_ref = min(epo_spec_ref_lap_avg_1.x(:,ind_lap(1),:));

max_max_2 = max(max(max_2,max_2_ref));
min_min_2 = min(min(min_2,min_2_ref));

% plot 
figure;subplot(1,2,1);plotChannel(epo_lap_spec,plot_chan_lap,'yLim',[min_min max_max]);
subplot(1,2,2);plotChannel(epo_spec_ref_lap,plot_chan_lap,'yLim',[min_min max_max]);

figure('Name','Average');subplot(1,2,1);plotChannel(epo_spec_lap_avg_1,plot_chan_lap{1},'yLim',[min_min_2 max_max_2]);
subplot(1,2,2);plotChannel(epo_spec_ref_lap_avg_1,plot_chan_lap{1},'yLim',[min_min_2 max_max_2]);
grid_markIval([Para.modfreq(1)-1 Para.modfreq(1)+1]);
grid_markIval([Para.modfreq(3)-1 Para.modfreq(3)+1]);
grid_markIval([Para.modfreq(5)-1 Para.modfreq(5)+1]);
grid_markIval([Para.modfreq(7)-1 Para.modfreq(7)+1]);
grid_markIval([Para.modfreq(9)-1 Para.modfreq(9)+1]);
for ii = 1:length(ind_lap)  % number of channels
    for i = 1:length(Para.modfreq)  % number of stimuli frequencies
        
        % Spectrum calculated after averaging
        epo_spec_lap_avg_1 = proc_spectrum(epo_lap_avg,[Para.modfreq(i)-1 Para.modfreq(i)+1], kaiser(cnt.fs,2));
        epo_spec_ref_lap_avg_1 = proc_spectrum(epo_ref_lap_avg,[Para.modfreq(i)-1 Para.modfreq(i)+1], kaiser(cnt.fs,2));
%         offset=min(min(min(epo_spec_lap_avg_1.x)));
%         offset_ref =min(min(min(epo_spec_ref_lap_avg_1.x)));
%         offset1=ceil(abs(min(offset,offset_ref)));
%         
        % Calculating the band Power after rescaling data
        epo_spec_lap_avg_band_all_1 = sum(epo_spec_lap_avg_1.x(:,ind_lap(ii),i));
        epo_spec_ref_lap_avg_band_all_1 = sum(epo_spec_ref_lap_avg_1.x(:,ind_lap(ii)));
        
        % Spectrum calculated before averaging the timeseries
%         epo_spec_lap = proc_spectrum(epo_lap,[Para.modfreq(i)-1 Para.modfreq(i)+1], kaiser(cnt.fs,2));
%         epo_spec_ref_lap = proc_spectrum(epo_ref_lap,[Para.modfreq(i)-1 Para.modfreq(i)+1], kaiser(cnt.fs,2));
%         
%         % Averaging the spectra
%         epo_spec_lap_avg = proc_average(epo_spec_lap,'classes',epo_spec_lap.className);
%         epo_spec_ref_lap_avg = proc_average(epo_spec_ref_lap);
%         
%         % Calculating the band Power after rescaling data
%         epo_spec_lap_avg_band_all = sum(10.^(epo_spec_lap_avg.x(:,ind_lap(ii),i)/10));
%         epo_spec_ref_lap_avg_band_all = sum(10.^(epo_spec_ref_lap_avg.x(:,ind_lap(ii))/10));

        % Calulating the relative increase in bandpower 
       % tuningfunc_1(ii,i) = (epo_spec_lap_avg_band_all_1/epo_spec_ref_lap_avg_band_all_1-1)*-100;%((epo_spec_lap_avg_band_all_1-epo_spec_ref_lap_avg_band_all_1)/abs(epo_spec_ref_lap_avg_band_all_1))*100;%((epo_spec_lap_avg_band_all_1(i)*100)/epo_spec_ref_lap_avg_band_all_1(i))-100;
        tuningfunc_1(ii,i) = ((epo_spec_lap_avg_band_all_1-epo_spec_ref_lap_avg_band_all_1)/abs(epo_spec_ref_lap_avg_band_all_1))*100;   
       % dif(ii,i) = tuningfunc_1(ii,i)-tuningfunc(ii,i);
        %         tuningfunc(ii,i) = ((epo_spec_lap_avg_band_all-epo_spec_ref_lap_avg_band_all)/epo_spec_ref_lap_avg_band_all)*100;%((epo_spec_lap_avg_band_all(i)*100)/epo_spec_ref_lap_avg_band_all(i))-100;

    end
end

% saving the analysed data
if saves ==1
save(['Tuningfunc/TuningFunc_',VP_CODE,file], 'tuningfunc_1');
end

%% Plots
% 
% %  Tuning Bars
% figure('Name','Contra-lateral (average after spec)')
% subplot(1,2,1);bar(Para.modfreq,tuningfunc(1,:));
% title(['Channel',plot_chan_lap{1}])
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
% ylim([min(min(tuningfunc))-50 max(max(tuningfunc))+50]);
% 
% subplot(1,2,2);bar(Para.modfreq,tuningfunc(2,:));
% title(['Channel',plot_chan_lap{2}])
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
% ylim([min(min(tuningfunc))-50 max(max(tuningfunc))+50]);

figure('Name','Contra-lateral (average before spec)')
subplot(1,2,1);bar(Para.modfreq,tuningfunc_1(1,:));
title(['Channel',plot_chan_lap{1}])
xlabel('Repetitive Frequency [Hz]')
ylabel('Bandpower increase [dB]')
ylim([min(min(tuningfunc_1))-50 max(max(tuningfunc_1))+50]);
subplot(1,2,2);bar(Para.modfreq,tuningfunc_1(2,:));
title(['Channel',plot_chan_lap{2}])
xlabel('Repetitive Frequency [Hz]')
ylabel('Bandpower increase [dB]')
ylim([min(min(tuningfunc_1))-50 max(max(tuningfunc_1))+50]);

% 
% figure('Name','Contra-lateral (dif)')
% subplot(1,2,1);bar(Para.modfreq,dif(1,:));
% title(['Channel',plot_chan_lap{1}])
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
% ylim([min(min(dif))-50 max(max(dif))+50]);
% subplot(1,2,2);bar(Para.modfreq,dif(2,:));
% title(['Channel',plot_chan_lap{2}])
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
% ylim([min(min(dif))-50 max(max(dif))+50]);

% 
% figure('Name',['Tuningfunction_lapdata',plot_chan_lap{1}]);
% plot(Para.modfreq(1:9),tuningfunc(1,:),'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
% grid on
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
% 
% figure('Name',['Tuningfunction_lapdata (average before spec)',plot_chan_lap{1}]);
% plot(Para.modfreq(1:9),tuningfunc_1(1,:),'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
% grid on
% xlabel('Repetitive Frequency [Hz]')
% ylabel('Bandpower increase [dB]')
%%
% 
% %% Produce figures
% save_path = 'C:\Documents and Settings\Min Konto\Dokumenter\Fouads Mappe\DTU\tiendesem\BCI-SSSEP Projekt\Matlab\master_fouad\';
% h_t = figure('Name','Epochs - Timeseries'); grid_plot(epo, mnt,opt_grid)
% if saving == 1
%     saveas(h_t,[save_path,VP_CODE,'_figs\Epochs - Timeseries'],'png')
% end
% h_raw=figure('Name','Epochs_raw - FFT');H=grid_plot(epo_spec, mnt,opt_spec);
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% 
% hgrid= grid_addBars(epo_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
% if saving == 1
%     saveas(h_raw,[save_path,VP_CODE,'_figs\Epochs_raw_FFT'],'png')
% end
% h_lap=figure('Name','Epochs_lap - FFT');H=grid_plot(epo_lap_spec,mnt_lap);
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% grid_addBars(epo_lap_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
% if saving==1
%     saveas(h_lap,[save_path,VP_CODE,'_figs\Epochs_lap - FFT'],'png')
% % end
% h_raw_full = figure('Name','Epochs_raw - FFT (Full Montage)');H=grid_plot(epo_spec,mnt1);
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% if saving==1
%     saveas(h_raw_full,[save_path,VP_CODE,'_figs\Epochs_raw - FFT (Full Montage)'],'png')
% end
% h_lap_full = figure('Name','Epochs_lap - FFT (Full Montage)');H=grid_plot(epo_lap_spec, mnt1);
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% if saving == 1
%     saveas(h_lap_full,[save_path,VP_CODE,'_figs\Epochs_lap - FFT (Full Montage)'],'png')
% end
% 
% figure;grid_plot(epo_spec_ref_lap,mnt1,opt_spec)
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% 
% figure;grid_plot(epo_spec_ref_lap,mnt,opt_spec)
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% 
% h_bi = figure('Name','Epochs_bipol - FFT');H=grid_plot(epo_bi_spec, mnt_bi,opt_spec);
% grid_markIval([modfreq-1 modfreq+1]);
% grid_markIval([modfreq1-1 modfreq1+1]);
% grid_markIval([modfreq2-1 modfreq2+1]);
% grid_markIval([modfreq3-1 modfreq3+1]);
% grid_markIval([modfreq4-1 modfreq4+1]);
% grid_markIval([modfreq5-1 modfreq5+1]);
% grid_markIval([modfreq6-1 modfreq6+1]);
% grid_markIval([modfreq7-1 modfreq7+1]);
% grid_markIval([modfreq8-1 modfreq8+1]);
% % grid_addBars(epo_spec_bi_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
% if saving ==1
%     saveas(h_bi,[save_path,VP_CODE,'_figs\Epochs_bipol - FFT'],'png')
% end
% h_ref = figure('Name','REF - FFT');H=grid_plot(epo_spec_ref, mnt,opt_spec);
% if saving==1
%     saveas(h_ref,[save_path,VP_CODE,'_figs\REF - FFT'],'png')
% end