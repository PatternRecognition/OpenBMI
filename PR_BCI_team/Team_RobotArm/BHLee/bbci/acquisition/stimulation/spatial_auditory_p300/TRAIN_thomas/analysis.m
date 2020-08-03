%% load the raw data

file='C:\data\bbciRaw\VPgao_09_11_25\VPgaoLoudVariFast';

% read the header
hdr= eegfile_readBVheader(file);

% design low-pass filter
Wps= [42 49]/hdr.fs*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 40);
[filt_eeg.b, filt_eeg.a]= cheby2(n, 50, Ws);


% load file excluding EMG channels:
[cnt, mrk]= eegfile_readBV([file '*'], 'fs',200, 'clab',{'not','EMG*'},'filt',filt_eeg);

%% remove artifacts

[mk, rClab]= reject_varEventsAndChannels(cnt, mrk, [-100,300], 'visualize',1);

