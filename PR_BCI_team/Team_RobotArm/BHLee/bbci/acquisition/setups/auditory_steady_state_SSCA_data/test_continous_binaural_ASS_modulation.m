% find lower and upper bounds of sound intensity

%% ASS base-sound params

scf = 500; % sound carrier frequency
ss_freq = [40, 40]; % steady state frequency in Hertz [left right]
sound_fs = 22050; % sampling frequency of the wave sound
% sound_fs = 44100; % sampling frequency of the wave sound

T_test = 2; % test tone duration in seconds

%% create test tone
test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', 1, 'fs', sound_fs, 'rampon', 20, 'rampoff', 50)';
% test_tone = randn(1,T_test*sound_fs)/2;
N = length(test_tone);
steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq(1)*T_test/N);
ss_test_tone = test_tone .* steady_state_modulation;

%% find lower sound amplitude bound
A_low_db = -45;
A_low = 10^(A_low_db/20);
wavplay(A_low*ss_test_tone,  sound_fs, 'async')

%% show fixed upper amplitude bound (relative to lower bound)
A_high_db = 30;
A_high = 10^(A_high_db/20) * A_low;
wavplay(A_high*ss_test_tone,  sound_fs, 'async')

%% generate slow amplitude modulation
T = 60*5; % time in seconds
modulation_cutoff_freq = 0.075;
fading_time = 10; % time in seconds
fading_window_end = 0.5 + 0.5*cos(linspace(0, pi, fading_time*sound_fs));
fading_window_start = fading_window_end(end:-1:1);

sm = create_slow_random_signal(T, sound_fs, modulation_cutoff_freq, 2);
sm = sm-repmat(mean(sm,2), 1, length(sm));
sm = sm ./repmat(std(sm,[],2)*2, 1, length(sm));
sm_db = (sm + 1) * A_high_db/2;

fading_window = [fading_window_start ones(1,length(sm)-2*length(fading_window_start)) fading_window_end];
fading_window = [fading_window; fading_window];
sm_db = sm_db.*fading_window;

clear sm fading_window

fs_downsample = 100;
N = length(sm_db);
N_ds = floor(N*(fs_downsample/sound_fs));
sm_db_downsampled = zeros(2,N_ds);
ds = round(linspace(1,N,N_ds));
for k=1:size(sm_db,1)
    sm_db_downsampled(k,:) = sm_db(k,ds);
end
figure
rows = 2;
cols = 1;
subplot(rows,cols,1)
plot((0:length(sm_db_downsampled)-1)/(fs_downsample), sm_db_downsampled')
ylabel('sound amplitude [dB]')
xlabel('time [seconds]')
title('sound intensity modulation in dB')
subplot(rows,cols,2)
plot((0:length(sm_db_downsampled)-1)/(fs_downsample), sm_db_downsampled'.^2)
title('SSCA target function - squared amplitude')
ylabel('sound amplitude [dB]')
xlabel('time [seconds]')

%% create stimulus
N = T*sound_fs;
sm_db = sm_db(:,1:N);
basis_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T * 1000, 'pan', 1, 'fs', sound_fs)';
steady_state_modulation = zeros(2,N);
ss_tone = zeros(2,N);
for k=1:2
    steady_state_modulation(k,:) = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq(k)*T/N);
    ss_tone(k,:) = basis_tone .* steady_state_modulation(k,:);
end
sm_ss_tone = ss_tone .* 10.^(sm_db/20) * A_low;

clear basis_tone ss_tone steady_state_modulation

figure, 
plot(sm_ss_tone')

%% play stimulus
wavplay(sm_ss_tone',  sound_fs, 'async')


%% stop the sound
clear playsnd;
