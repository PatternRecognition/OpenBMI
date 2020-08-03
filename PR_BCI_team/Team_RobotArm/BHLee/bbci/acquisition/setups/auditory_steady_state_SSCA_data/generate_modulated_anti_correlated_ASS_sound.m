function [sm_ss_tone, sm_db_downsampled] = generate_modulated_anti_correlated_ASS_sound(varargin)
%
%
% T - duration, given in seconds
% carrier_freq - carrier frequency of the tone (can be a vector of length 2)
% ss_freq - steady state frequency (can be a vector of length 2)
% sound_fs - sampling rate of the tone
% lp_freq - lowpass cutoff for slow amplitude modulation
% db_max - upper limit of loudness, given in dezibel relative to db_min
% db_min - lower limit of loudness, given in dezibel
% fs_downsample - sampling rate for down-sampled version of the slow
% amplitude modulation


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
    ,'T', 3*60 ...
    ,'carrier_freq', 1000 ...
    ,'ss_freq', 40 ...
    ,'sound_fs', 44100 ...
    ,'lp_freq', 0.075 ...
    ,'db_max', 30 ...
    ,'db_min', -30 ...
    ,'fs_downsample', 1000 ... 
);

T = opt.T;
scf = opt.carrier_freq;
ss_freq = opt.ss_freq;
sound_fs = opt.sound_fs;
modulation_cutoff_freq = opt.lp_freq;
A_high_db = opt.db_max;
A_low_db = opt.db_min;
fs_downsample = opt.fs_downsample; 

scf = two_element_row_vector(scf);
ss_freq = two_element_row_vector(ss_freq);
A_low_db = two_element_row_vector(A_low_db);
A_high_db = two_element_row_vector(A_high_db);


%% fixed params
fading_time = 10; % interval in which the tone is faded in and out at the beginning and at the end
A_low = 10.^(A_low_db/20);

%% generate slow amplitude modulation

% create the slow modulation (sm) signal
sm = create_slow_anti_correlated_random_signal(T, sound_fs, modulation_cutoff_freq);
% remove mean and rescale slow oscillation
sm = sm - repmat(min(sm,[],2), 1, length(sm));
sm = sm ./ repmat(max(sm,[],2), 1, length(sm));
sm_db = sm .* repmat(A_high_db', 1, length(sm));

clear sm

% downsample the slow modulation function
N = length(sm_db);
N_ds = floor(N*(fs_downsample/sound_fs));
sm_db_downsampled = zeros(2,N_ds);
ds = round(linspace(1,N,N_ds));
sm_db_downsampled = sm_db(:,ds);

%% create stimulus
N = T*sound_fs;
sm_db = sm_db(:,1:N);
    
ss_tone = zeros(2,N);
for k=1:2
    basis_tone = stimutil_generateTone(scf(k), 'harmonics',1, 'duration', T * 1000, 'pan', 1, 'fs', sound_fs)';
    steady_state_modulation = 0.5 + 0.5*sin((1:N)*2*pi*ss_freq(k)*T/N);
    ss_tone(k,:) = basis_tone .* steady_state_modulation;
end
sm_ss_tone = ss_tone .* 10.^(sm_db/20) .* repmat(A_low', [1,length(ss_tone)]);

function x = two_element_row_vector(x)
% turn x into a row vector
if length(x)==1
    x = x*ones(1,2);
end
if size(x,1) > 1
    x = x';
end
if not(size(x,1)==1 && size(x,2)==2)
    error('x is not a two element row vector!!!');
end