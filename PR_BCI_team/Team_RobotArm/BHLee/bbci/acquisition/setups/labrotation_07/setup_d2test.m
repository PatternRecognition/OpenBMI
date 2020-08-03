% no need to run following 2 commands in real recording
%acq_getDataFolder('tmp')
%setup_bbci_bet_unstable

% bbciclose

% to calculate experiment total length
% (opt.duration_cue + opt.response_delay(1) + opt.duration_response + opt.duration_blank) * N / 60000

% ppTrigger(100) - stimuli off
% ppTrigger(101) - reactometer on
% ppTrigger(102) - reactometr off

% opt.duration_cue - how long the cue will bw presented
% opt.response_delay - how long we need to wait, after the stimuli is presented, until the response is shown
% opt.duration_response - how long the reactometer is shown
% opt.duration_blank - how long there will be blank screen, after
% reactometer is off (before new stimuli)

N= 250;

opt= struct('perc_dev', 0.5);
opt.response_markers= {'R 16', 'R  8'};
opt.bv_host= 'localhost';
opt.position = [-1919 0 1920 1181];
%opt.position= [5 500 640 480];
opt.duration_cue=  350;
opt.response_delay= 400*[1 1];
opt.duration_response= 350;
opt.duration_blank= 350;
%opt.test = 1;

fprintf('for testing:\n  stim_d2test(30, opt, ''test'', 1);\n');
fprintf('stim_d2test(N, opt);\n');
