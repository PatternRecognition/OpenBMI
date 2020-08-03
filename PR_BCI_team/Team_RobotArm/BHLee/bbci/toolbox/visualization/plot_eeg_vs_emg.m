function [errEEG,errEMG]= plot_eeg_vs_emg(cnt, mrk, E, eeg_proc, emg_proc);
%[errEEG,errEMG]= plot_eeg_vs_emg(cnt, mrk, eeg_proc, emg_proc, model, E);
%
% C validate_over_time

if ~exist('emg_proc', 'var'), emg_proc= 'fv= proc_detectEMG(epo);'; end
if ~isstruct(eeg_proc), eeg_proc= struct('proc',eeg_proc); end
if ~isfield(eeg_proc, 'chans'), eeg_proc.chans= {'not','E*'}; end
if ~isfield(eeg_proc, 'model'), eeg_proc.model= 'LDA'; end
if ~isfield(eeg_proc, 'xTrials'), eeg_proc.xTrials= [10 10]; end
if ~isfield(eeg_proc, 'ilen'), eeg_proc.ilen= 1270; end
if ~isstruct(emg_proc), emg_proc= struct('proc',emg_proc); end
if ~isfield(emg_proc, 'chans'), emg_proc.chans= {'EMG*'}; end
if ~isfield(emg_proc, 'model'), emg_proc.model= 'LDA'; end
if ~isfield(emg_proc, 'xTrials'), emg_proc.xTrials= eeg_proc.xTrials; end
if ~isfield(emg_proc, 'ilen'), emg_proc.ilen= 200; end

global NO_CLOCK
no_clock_memo= NO_CLOCK;
NO_CLOCK= 1;

errEEG= validate_over_time(cnt, mrk, E, eeg_proc);
errEMG= validate_over_time(cnt, mrk, E, emg_proc);
errEOG= validate_over_time(cnt, mrk, E, emg_proc);

clf;
axes('position', [0.15 0.2 0.7 0.7]);
hp= plot(E, [errEEG(:,1) errEMG(:,1)], 'lineWidth',2);
set(hp(2), 'color','r');
set(gca, 'xLim', E([2 end-1]), 'yLim',[0 52], ...
         'yTick',0:10:50, 'box','off');
legend('EEG', 'EMG', 3);
xlabel('point in time of causal classification [ms]');
ylabel('validation error [%]');
grid on
title(untex(cnt.title));

NO_CLOCK= no_clock_memo;
