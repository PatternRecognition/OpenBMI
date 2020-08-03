%% --- initialization --- 

% make sure VP_CODE is set and the setup script has been executed!

bvr_sendcommand('viewsignals');


%% --- auditory oddball --- 

interval_standard_1 = 1000; % ISI, milliseconds
duration_standard_1 = 3;    % duration of experiment, in minutes
rate = 5;                   % Non-Target ratio: 1/rate

opt = [];
opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

opt.isi_jitter = 0; % defines jitter in ISI
opt.speech_intro = '';
opt.fixation = 1;
opt.require_response = 0;
opt.fs = 44100;
opt.cue_std = stimutil_generateTone(500, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_std = opt.cue_std*.25;
opt.cue_dev = stimutil_generateTone(1000, 'harmonics', 7, 'duration', 50, 'pan', 1, 'fs', opt.fs);
opt.cue_dev = opt.cue_dev*.25;

sequenz = standard_oddball(rate,interval_standard_1,duration_standard_1);
sequenz  = sequenz - 1;

fprintf('Standardmessung 1 starten');
opt.isi = 1000;
opt.filename = ['auditory_oddball_isi_' num2str(opt.isi) '_' VP_CODE];
opt.impedances = 0;
opt.test = 0; % test = 1 means no recording!!!
sprintf('press <RETURN> to proceed with the standard Auditory Oddball Experiment with ISI: %d', opt.isi)
pause;
play_auditory_oddball_ISI(sequenz, opt);
sprintf('how many did you count (TRUE NUMBER: %i, ISI %i)?\n',sum(sequenz), opt.isi)

%%  --- steady-state-stuff --- 
%% ASS base-sound params

scf = 500; % sound carrier frequency
ss_freq = 40; % steady state frequency in Hertz [left right]
sound_fs = 22050; % sampling frequency of the wave sound
% sound_fs = 44100; % sampling frequency of the wave sound

T_test = 2; % test tone duration in seconds

% create test tone
test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', [1,1], 'fs', sound_fs, 'rampon', 20, 'rampoff', 50)';
N = length(test_tone);
steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq(1)*T_test/N);
ss_test_tone = test_tone .* repmat(steady_state_modulation, size(test_tone,1), 1);


%% show sound stimulus
% play the sound stimulus a couple of times to familiarize the subject with
% it. try to determine a suitable start value for hearing-threshold
% procedure. it should be clearly above the threshold but not too far above

A_test_db = -50; % play around with this value
A_test = 10^(A_test_db/20);
wavplay(A_test*ss_test_tone',  sound_fs, 'async')


%% determine hearing threshold -- left ear
pan = [1,0];

test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', pan, 'fs', sound_fs, 'rampon', 20, 'rampoff', 50)';
N = length(test_tone);
ss_test_tone = test_tone .* repmat(steady_state_modulation, size(test_tone,1), 1);

th_opt = [];
th_opt.test_tone = ss_test_tone;
th_opt.fs = sound_fs;
th_opt.db_start = A_test_db;
th_opt.delta_dB = 2;
display('Type <ENTER> to start determining hearing threshold')
pause
[A_th_db_left, foo, db_values_left] = determine_hearing_threshold_via_staircase(th_opt);

figure,
plot(db_values_left)
hold on
plot(ones(1,length(db_values_left))*A_th_db_left, 'r')
ylabel('sound amplitude [dB]');


%% determine hearing threshold -- right ear
pan = [0,1];

test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', pan, 'fs', sound_fs, 'rampon', 20, 'rampoff', 50)';
N = length(test_tone);
ss_test_tone = test_tone .* repmat(steady_state_modulation, size(test_tone,1), 1);

th_opt = [];
th_opt.test_tone = ss_test_tone;
th_opt.fs = sound_fs;
th_opt.db_start = A_test_db;
th_opt.delta_dB = 2;
display('Type <ENTER> to start determining hearing threshold')
pause
[A_th_db_right, foo, db_values_right] = determine_hearing_threshold_via_staircase(th_opt);

figure,
plot(db_values_right)
hold on
plot(ones(1,length(db_values_right))*A_th_db_right, 'r')
ylabel('sound amplitude [dB]');



%% show test tone with ear-specfic baselines
% the test tone should appear equally loud on both ears
A_low_db = 10 + [A_th_db_left, A_th_db_right];
A_low = 10.^(A_low_db/20);

test_tone = stimutil_generateTone(scf, 'harmonics',1, 'duration', T_test * 1000, 'pan', [1,1], 'fs', sound_fs, 'rampon', 20, 'rampoff', 50)';
N = length(test_tone);
ss_test_tone = test_tone .* repmat(steady_state_modulation, size(test_tone,1), 1);

wavplay(repmat(A_low,[length(ss_test_tone),1]).*ss_test_tone',  sound_fs, 'async')

%% show fixed upper amplitude bound (relative to lower bound)
% the upper bound should be 30 or 35 db above the lower bound. it should
% not be perceived as too loud however! if it is too loud, the speakers or
% earphones may saturate or produce strange artifacts (like strange
% harmonics...) 
A_high_db = 35*ones(1,2); % perhaps raise it slowly. at the end it should be fixed to 30
A_high = 10.^(A_high_db/20) .* A_low;
wavplay(repmat(A_high,[length(ss_test_tone),1]).*ss_test_tone',  sound_fs, 'async')



%% --- Recording 1: One modulation for both ears ----

test = 1; % test = 1 means no recording!!!
if test && not(exist('TODAY_DIR', 'var'))
    TODAY_DIR = fullfile(EEG_RAW_DIR, 'VPtest');
end

start_marker = 251;
end_marker = 254;

file_prefix = 'ASS_eyesOpen';

params = [];
params.n_runs = 3;
params.T = 5*60;
params.carrier_freq = 500;
params.ss_freq = 40;
params.sound_fs = 22050;
params.lp_freq = 0.05; 
params.db_max = A_high_db;
params.db_min = A_low_db;
params.fs_downsample = 1000;
params.same_sound = 1;

for k=1:params.n_runs
    
    fprintf('Run %d: Press <ENTER> to start stimulus generation\n', k);
    pause

    display('generating stimulus')
    [sm_ss_tone, sm_db_downsampled] = generate_modulated_ASS_sound(params);
    display('done')
    % plot the stimulus
    figure
    rows = 4;
    cols = 1;
    time_vec = (1:length(sm_db_downsampled))/params.fs_downsample;
    subplot(rows,cols,1)
    plot(time_vec, sm_db_downsampled')
    title('sound modulation')
    ylabel('[db]')
    xlabel('time [s]')
    subplot(rows,cols,2)
    plot(sm_ss_tone')
    title('sound')
    subplot(rows,cols,3)
    plot(time_vec, sm_db_downsampled'.^2)
    title('expected power modulation')
    ylabel('[a.u.]')
    xlabel('time [s]')
    subplot(rows,cols,4)
    hist(sm_db_downsampled, 50)
    title('histogram of power modulation')
    xlabel('[db]')


    % start the eeg recording and stimulus presentation
    display('Press <ENTER> to start the trial.')
    pause
    
    display('Starting the trial!')
    if not(test)
        bvr_startrecording([file_prefix '_' VP_CODE], 'impedances', 0);
    end
    pause(10);
    ppTrigger(start_marker)
    if not(test)
        wavplay(sm_ss_tone',  params.sound_fs, 'sync')
    else
        wavplay(sm_ss_tone',  params.sound_fs, 'async')
        display('Press <ENTER> to continue')
        pause
    end
    ppTrigger(end_marker)
    pause(5)
    if not(test)
        bvr_sendcommand('stoprecording');
    end
    
    % save parameters and stimulus
    display('Saving stimulus parameters and modulation function')
    if not(exist(TODAY_DIR, 'dir'))
        mkdir(TODAY_DIR);
    end
    fname = fullfile(TODAY_DIR, sprintf('%s_%s_params_and_modulation_function_%02d.mat', file_prefix, VP_CODE, k));
    save(fname, 'params', 'sm_db_downsampled')
    wav_name = fullfile(TODAY_DIR, sprintf('%s_%s_auditory_stimulus_%02d', file_prefix, VP_CODE, k));
    wavwrite(sm_ss_tone', params.sound_fs, wav_name);
    display('done saving')
    
    clear sm_ss_tone
end

%% stop the sound if necessary
clear playsnd;

%% --- Recording 2: Ear-specific modulation ----

test = 0; % test = 1 means no recording!!!
if test && not(exist('TODAY_DIR', 'var'))
    TODAY_DIR = fullfile(EEG_RAW_DIR, 'VPtest');
end

start_marker = 251;
end_marker = 254;

file_prefix = 'ASS_eyesOpen_earSpecificModulation';

params = [];
params.n_runs = 2;
params.T = 5*60;
params.carrier_freq = 500;
params.ss_freq = [40, 40];
params.sound_fs = 22050;
params.lp_freq = 0.05; 
params.db_max = A_high_db;
params.db_min = A_low_db;
params.fs_downsample = 1000;
params.same_sound = 0;

for k=1:params.n_runs

    display('Press <ENTER> to start stimulus generation')
    pause

    % generate stimulus
    display('generating stimulus')
    [sm_ss_tone, sm_db_downsampled] = generate_modulated_ASS_sound(params);
    display('done')
    % plot the stimulus
    figure
    rows = 3;
    cols = 1;
    time_vec = (1:length(sm_db_downsampled))/params.fs_downsample;
    subplot(rows,cols,1)
    plot(time_vec, sm_db_downsampled')
    title('sound modulation')
    ylabel('[db]')
    xlabel('time [s]')
    subplot(rows,cols,2)
    plot(sm_ss_tone')
    title('sound')
    subplot(rows,cols,3)
    plot(time_vec, sm_db_downsampled'.^2)
    title('expected power modulation')
    ylabel('[a.u.]')
    xlabel('time [s]')


    % start the eeg recording and stimulus presentation
    display('Press <ENTER> to start the trial.')
    pause
    
    display('Starting the trial!')
    if not(test)
        bvr_startrecording([file_prefix '_' VP_CODE], 'impedances', 0);
    end
    pause(10);
    ppTrigger(start_marker)
    if not(test)
        wavplay(sm_ss_tone',  params.sound_fs, 'sync')
    else
        wavplay(sm_ss_tone',  params.sound_fs, 'async')
        display('Press <ENTER> to continue')
        pause
    end
    ppTrigger(end_marker)
    pause(5)
    if not(test)
        bvr_sendcommand('stoprecording');
    end
    
    % save parameters and stimulus
    display('Saving stimulus parameters and modulation function')
    if not(exist(TODAY_DIR, 'dir'))
        mkdir(TODAY_DIR);
    end
    fname = fullfile(TODAY_DIR, sprintf('%s_%s_params_and_modulation_function_%02d.mat', file_prefix, VP_CODE, k));
    save(fname, 'params', 'sm_db_downsampled')
    wav_name = fullfile(TODAY_DIR, sprintf('%s_%s_auditory_stimulus_%02d', file_prefix, VP_CODE, k));
    wavwrite(sm_ss_tone', params.sound_fs, wav_name);
    display('done saving')
    
    clear sm_ss_tone
end

%% stop the sound if necessary
clear playsnd;

%% --- Recording 3: Ear-specific modulation and steady-state frequency ----

test = 0; % test = 1 means no recording!!!
if test && not(exist('TODAY_DIR', 'var'))
    TODAY_DIR = fullfile(EEG_RAW_DIR, 'VPtest');
end

start_marker = 251;
end_marker = 254;

file_prefix = 'ASS_eyesOpen_earSpecificModulationAndSSFrequency';

params = [];
params.n_runs = 2;
params.T = 5*60;
params.carrier_freq = 500;
params.ss_freq = [37, 67];
params.sound_fs = 22050;
params.lp_freq = 0.05; 
params.db_max = A_high_db;
params.db_min = A_low_db;
params.fs_downsample = 1000;
params.same_sound = 0;

for k=1:params.n_runs
    
    display('Press <ENTER> to start stimulus generation')
    pause
    
    % generate stimulus
    display('generating stimulus')
    [sm_ss_tone, sm_db_downsampled] = generate_modulated_ASS_sound(params);
    display('done')
    % plot the stimulus
    figure
    rows = 3;
    cols = 1;
    time_vec = (1:length(sm_db_downsampled))/params.fs_downsample;
    subplot(rows,cols,1)
    plot(time_vec, sm_db_downsampled')
    title('sound modulation')
    ylabel('[db]')
    xlabel('time [s]')
    subplot(rows,cols,2)
    plot(sm_ss_tone')
    title('sound')
    subplot(rows,cols,3)
    plot(time_vec, sm_db_downsampled'.^2)
    title('expected power modulation')
    ylabel('[a.u.]')
    xlabel('time [s]')


    % start the eeg recording and stimulus presentation
    display('Press <ENTER> to start the trial.')
    pause
    
    display('Starting the trial!')
    if not(test)
        bvr_startrecording([file_prefix '_' VP_CODE], 'impedances', 0);
    end
    pause(10);
    ppTrigger(start_marker)
    if not(test)
        wavplay(sm_ss_tone',  params.sound_fs, 'sync')
    else
        wavplay(sm_ss_tone',  params.sound_fs, 'async')
        display('Press <ENTER> to continue')
        pause
    end
    ppTrigger(end_marker)
    pause(5)
    if not(test)
        bvr_sendcommand('stoprecording');
    end
    
    % save parameters and stimulus
    display('Saving stimulus parameters and modulation function')
    if not(exist(TODAY_DIR, 'dir'))
        mkdir(TODAY_DIR);
    end
    fname = fullfile(TODAY_DIR, sprintf('%s_%s_params_and_modulation_function_%02d.mat', file_prefix, VP_CODE, k));
    save(fname, 'params', 'sm_db_downsampled')
    wav_name = fullfile(TODAY_DIR, sprintf('%s_%s_auditory_stimulus_%02d', file_prefix, VP_CODE, k));
    wavwrite(sm_ss_tone', params.sound_fs, wav_name);
    display('done saving')
    
    clear sm_ss_tone
end

%% stop the sound if necessary
clear playsnd;

%% --- Recording 4: Ear-specific anti-correlated modulation and steady-state frequency ----

test = 0; % test = 1 means no recording!!!
if test && not(exist('TODAY_DIR', 'var'))
    TODAY_DIR = fullfile(EEG_RAW_DIR, 'VPtest');
end

start_marker = 251;
end_marker = 254;

file_prefix = 'ASS_eyesOpen_earSpecificAntiCorrelatedModulationAndSSFrequency';

params = [];
params.n_runs = 2;
params.T = 5*60;
params.carrier_freq = 500;
params.ss_freq = [37, 67];
params.sound_fs = 22050;
params.lp_freq = 0.05; 
params.db_max = A_high_db;
params.db_min = A_low_db;
params.fs_downsample = 1000;

for k=1:params.n_runs
    
    display('Press <ENTER> to start stimulus generation')
    pause
    
    % generate stimulus
    display('generating stimulus')
    [sm_ss_tone, sm_db_downsampled] = generate_modulated_anti_correlated_ASS_sound(params);
    display('done')
    % plot the stimulus
    figure
    rows = 3;
    cols = 1;
    time_vec = (1:length(sm_db_downsampled))/params.fs_downsample;
    subplot(rows,cols,1)
    plot(time_vec, sm_db_downsampled')
    title('sound modulation')
    ylabel('[db]')
    xlabel('time [s]')
    subplot(rows,cols,2)
    plot(sm_ss_tone')
    title('sound')
    subplot(rows,cols,3)
    plot(time_vec, sm_db_downsampled'.^2)
    title('expected power modulation')
    ylabel('[a.u.]')
    xlabel('time [s]')


    % start the eeg recording and stimulus presentation
    display('Press <ENTER> to start the trial.')
    pause
    
    display('Starting the trial!')
    if not(test)
        bvr_startrecording([file_prefix '_' VP_CODE], 'impedances', 0);
    end
    pause(10);
    ppTrigger(start_marker)
    if not(test)
        wavplay(sm_ss_tone',  params.sound_fs, 'sync')
    else
        wavplay(sm_ss_tone',  params.sound_fs, 'async')
        display('Press <ENTER> to continue')
        pause
    end
    ppTrigger(end_marker)
    pause(5)
    if not(test)
        bvr_sendcommand('stoprecording');
    end
    
    % save parameters and stimulus
    display('Saving stimulus parameters and modulation function')
    if not(exist(TODAY_DIR, 'dir'))
        mkdir(TODAY_DIR);
    end
    fname = fullfile(TODAY_DIR, sprintf('%s_%s_params_and_modulation_function_%02d.mat', file_prefix, VP_CODE, k));
    save(fname, 'params', 'sm_db_downsampled')
    wav_name = fullfile(TODAY_DIR, sprintf('%s_%s_auditory_stimulus_%02d', file_prefix, VP_CODE, k));
    wavwrite(sm_ss_tone', params.sound_fs, wav_name);
    display('done saving')
    
    clear sm_ss_tone
end

%% stop the sound if necessary
clear playsnd;