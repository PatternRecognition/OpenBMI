% find lower and upper bounds of sound intensity

%% ASS base-sound params

scf = 500; % sound carrier frequency
ss_freq = 40; % steady state frequency in Hertz
% sound_fs = 44100; % sampling frequency of the wave sound
sound_fs = 22050; % sampling frequency of the wave sound

T_test = 2; % test tone duration in seconds

%% create test tone
test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', 1, 'fs', sound_fs, 'rampon', 20, 'rampoff', 20)';
% test_tone = randn(1,T_test*sound_fs)/2;
N = length(test_tone);
steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq*T_test/N);
ss_test_tone = test_tone .* steady_state_modulation;

%% find lower sound amplitude bound
A_low_db = -25;
A_low = 10^(A_low_db/20);
wavplay(A_low*ss_test_tone,  sound_fs, 'async')

%% show fixed upper amplitude bound (relative to lower bound)
A_high_db = 25;
A_high = 10^(A_high_db/20) * A_low;
wavplay(A_high*ss_test_tone,  sound_fs, 'async')

%% generate slow amplitude modulation
T = 60*5; % time in seconds
modulation_cutoff_freq = 0.05;
fading_time = 10; % time in seconds
fading_window_end = 0.5 + 0.5*cos(linspace(0, pi, fading_time*sound_fs));
fading_window_start = fading_window_end(end:-1:1);

sm = create_slow_random_signal(T, sound_fs, modulation_cutoff_freq);
sm = sm-mean(sm);
fading_window = [fading_window_start ones(1,length(sm)-2*length(fading_window_start)) fading_window_end];
sm = sm.*fading_window;
sm = sm /(std(sm)*2);

sm_db = (sm + 1) * A_high_db/2;

figure
plot((0:length(sm_db)-1)/(sound_fs), sm_db)
ylabel('sound amplitude [dB]')
xlabel('time [seconds]')

%% create stimulus
basis_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T * 1000, 'pan', 1, 'fs', sound_fs, 'rampon', 20, 'rampoff', 20)';
N = length(basis_tone);
steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq*T/N);
ss_tone = basis_tone .* steady_state_modulation;
sm_db = sm_db(1:N);
sm_ss_tone = ss_tone .* 10.^(sm_db/20) * A_low;

%% play stimulus
wavplay(sm_ss_tone,  sound_fs, 'async')


%% stop the sound
clear playsnd;
