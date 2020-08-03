%DEMO_VISUALIZE_ERPS
%
%Description:
% This demo shows an example of how to visualize Event-Related Potentials
% (ERPs), which are components of brain-signals that are non-oscillatory
% and phase-locked to some (internal or external) event.
% The demo uses as example the P3 component of an odd-ball experiment.

% Author(s): Benjamin Blankertz, Feb 2005

file= [DATA_DIR 'siemensMat/Ben_05_01_07/einzelobjekteBen'];
[cnt, Mrk, mnt]= loadProcessedEEG(file, 'display');

%% First, plot the stimulus-aligned ERPs. Markers of stimuli are in
%% the substructure 'trg'.
mrk= Mrk.trg;

%% Define some channel layout.
grd= sprintf('EOGh,legend,Fz,scale,EOGv\nT7,CP3,CPz,CP4,T8\nTP7,P3,Pz,P4,TP8\nP7,PO3,POz,PO4,P8');
mnt= mnt_setGrid(mnt, grd);

%% Cut out segments (short-time windows) from the continuous signals
%% around each marker. Such segments of brain-signals are called epochs.
%% Choose epochs from 200ms before marker to 800ms after marker.
epo= makeEpochs(cnt, mrk, [-200 800]);
%% Subtract from each epochs the average value of the interval -200 to
%% 0ms (separately for each channel).
epo= proc_baseline(epo, [-200 0]);
%% Calculate classwise averages across trials.
erp= proc_average(epo);
%% Do the plot.
H= grid_plot(erp, mnt);

%% To verify the 'significance' of the difference between the two conditions
%% ... to be explained ...
epo_rsq= proc_r_square(epo);
%% and include the information into the plot. Since we want to have the
%% scale of the r^2-values, we pass the handle of the scale subplot.
grid_addBars(epo_rsq, 'h_scale',H.scale);

fprintf('ERPs shown as time series for some selected channels.\n');
fprintf('Press a key to continue.\n');
pause;


scalpPatternsPlusChannel(erp, mnt, 'Pz', [450 550; 400 500]);
fprintf('Scalp topographies of P3 components in both conditions.\n');
fprintf('Press a key to continue.\n');
pause;


scalpEvolutionPlusChannel(erp, mnt, 'Pz', 300:75:600);
fprintf('ERPs shown as scalp topographies in their evolution over time.\n');
