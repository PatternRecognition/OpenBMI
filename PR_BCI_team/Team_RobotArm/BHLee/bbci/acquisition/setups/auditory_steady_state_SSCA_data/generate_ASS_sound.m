%% generate slow changing ASS sound

T = 60*1.5; % time in seconds
ss_freq = 40; % steady state frequency in Hertz
sound_fs = 44100; % sampling frequency of the wave sound
modulation_cutoff_freq = 0.025;


display('generating ASS sound')
long_sound = stimutil_generateTone(500, 'harmonics',1, 'duration', 1000 * T, 'pan', 1, 'fs', sound_fs)';
% long_sound = randn(1,T*sound_fs);
N = length(long_sound);
steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq*T/N);
ss_sound = long_sound.*steady_state_modulation;
display('done')

display('generating slow amplitude modulation')
tmp_fs = 100;
sm = create_slow_random_signal(T, sound_fs, modulation_cutoff_freq);
sm = sm(1:length(ss_sound));

% normalize between -10 and 10 dB
sm = sm-min(sm);
sm = sm/max(sm);
sm = 20*sm - 10;
f = 10.^(sm/10);



display('done');


% sm_log
modulated_ss_sound = ss_sound .* f;

%% plot stuff

N = sound_fs * T;
time = (1:N)/sound_fs;
figure
rows = 2;
cols = 1;
subplot(rows,cols,1)
plot(time, sm)
title('Amplitude modulation of steady-state sound in dB units')
xlabel('time in seconds')
ylabel('dB')
subplot(rows,cols,2)
plot(time, f)
title('Amplitude modulation of steady-state sound absolute units')
xlabel('time in seconds')
ylabel('a.u.')
pause(1)

%% play the sound

wavplay(modulated_ss_sound,  sound_fs, 'async')

%% stop the sound
clear playsnd;

