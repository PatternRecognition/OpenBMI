clc;
warning('off', 'all');
fprintf('Welcome to the spatial auditory setup.\n');
if ~exist('VP_CODE', 'var'),
    error('VP_CODE has not been set.');
end

bvr_sendcommand('loadworkspace', 'martijns_study_alpha');
bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

fprintf('Ask the subject to fill out the form. Press <RETURN> when finished.\n');
pause

fprintf('Now running the speaker calibration.\n');
setup_spatialbci_common;
speaker_calibration(opt);
pause


SESSION_TYPE = 'OddballCounting';
setup_spatialbci_common;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballCountingFast';
setup_spatialbci_common;
opt.isi =  175;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballSingleCounting';
setup_spatialbci_common;
opt.singleSpeaker = 1;
trials = 10;
fprintf('Press <RETURN> when ready to start the first round of control stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballCounting';
setup_spatialbci_common;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballCountingFast';
setup_spatialbci_common;
opt.isi =  175;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballSingleCounting';
setup_spatialbci_common;
opt.singleSpeaker = 1;
trials = 10;
fprintf('Press <RETURN> when ready to start the first round of control stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballCounting';
setup_spatialbci_common;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);

pause
SESSION_TYPE = 'OddballCountingFast';
setup_spatialbci_common;
opt.isi =  175;
fprintf('Press <RETURN> when ready to start the first round of spatial stimuli.\n');
pause
stim_oddballAuditorySpatial(N, trials, opt);