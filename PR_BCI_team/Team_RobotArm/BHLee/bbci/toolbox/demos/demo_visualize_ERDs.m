%DEMO_VISUALIZE_ERPS
%
%Description:
% This demo shows an example of how to visualize Event-Related 
% (De)synchronization (ERD) curves, which are variations of the
% spectral energy in a specified frequency band over time.
% The demo uses as example motor imagery from the training session
% of a BCI experiment.

% Author(s): Benjamin Blankertz, Feb 2005

file= strcat('Klaus_04_04_08/imag_', {'lett','move'}, 'Klaus');
[cnt,mrk,mnt]= eegfile_loadMatlab(file);

%% Define some channel layout.
grd= sprintf('EOGh,legend,Fz,scale,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\n\nEMGl,P3,Pz,P4,EMGr');
mnt= mnt_setGrid(mnt, grd);

%% First plot classwise averaged spectra to find out what could be an
%% interesting frequency band.
epo= makeEpochs(cnt, mrk, [-500 4000]);

%% Apply a (spatial) laplace filter in order to have more localized
%% brain signals. Otherwise the spectral differences are only very weak.
epo= proc_laplace(epo);

%% Calculate spectra in the frequency range 5 to 35Hz.
spec= proc_spectrum(epo, [5 35]);

%% Do the plot.
grid_plot(spec, mnt, 'xTick', [10 20 30]);

%% Define the frequency band.
band= [8 25];
%% And mark it in the spectrum plot
grid_markIval(band);

fprintf('Press a key to continue.\n');
pause;



%% Apply a band-pass filter to the continuous EEG signals.
[b,a]= butter(5, band/cnt.fs*2);
cnt_flt= proc_channelwise(cnt, 'filtfilt', b, a);

%% Cut out segments (short-time windows) from the continuous signals
%% around each marker. Such segments of brain-signals are called epochs.
%% Choose epochs from 500ms before marker to 4000ms after marker.
epo= makeEpochs(cnt_flt, mrk, [-500 4000]);

%% Apply a (spatial) laplace filter in order to have more localized
%% brain signals. Otherwise the spectral differences are only very weak.
%% Specifying 'E*' as last argument copies all channels, whose labels
%% match 'E*' (i.e., 'EMGl','EMGr','EOGh','EOGv') without filtering.
epo= proc_laplace(epo, 'small', ' lap', 'E*');

%% In order to get the non-phase-locked signal changes, one has to
%% rectify the signals.
epo= proc_rectifyChannels(epo);

%% To get smoother curves, apply an moving average low-pass filter.
epo= proc_movingAverage(epo, 200, 'centered');

%% Subtract from each epochs the average value of the pre-stimulus
%% interval -500 to 0ms (separately for each channel).
epo= proc_baseline(epo, [-500 0]);

%% Calculate classwise averages.
erd= proc_average(epo);

%% Do the plot.
grid_plot(erd, mnt);

fprintf('ERDs shown as time series for some selected channels.\n');
fprintf('Press a key to continue.\n');
pause;


%% Plot scalp topographies of ERDs.
%% We repeat the same processing steps as above, just leaving out the
%% laplace filter.
epo= makeEpochs(cnt_flt, mrk, [-500 4000]);
epo= proc_rectifyChannels(epo);
epo= proc_movingAverage(epo, 200, 'centered');
epo= proc_baseline(epo, [-500 0]);
erd= proc_average(epo);

scalpEvolutionPlusChannel(erd, mnt, 'CP2', 1000:750:4000);
fprintf('ERDs shown as scalp topographies in their evolution over time.\n');
fprintf('The topographies do not look very specific, since they are\n');
fprintf('dominated by the parietal alpha rhythm.\n');
fprintf('Press a key to continue.\n');
pause;

%% Subtract the overall average
erd_ref= proc_classMean(erd);
erd_ref.className= {'mean'};
erd= proc_subtractReferenceClass(erd, erd_ref);

%% And repeat the plot.
scalpEvolutionPlusChannel(erd, mnt, 'CP2', 1000:750:4000);
fprintf('ERD differences (class-average minus overall average) shown\n');
fprintf('as scalp topographies in their evolution over time.\n');
