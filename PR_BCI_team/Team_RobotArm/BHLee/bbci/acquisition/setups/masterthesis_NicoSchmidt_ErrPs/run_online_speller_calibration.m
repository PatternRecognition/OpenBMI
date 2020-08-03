%% Prepare offline calibration

phrase= {'WINKT_','QUARZ_','FJORD'};
phrase = [phrase{:}];
phrase_practice = 'BCI';
offline_mode = 1;
do_ErrP_detection = 0;
nr_sequences = bbci.func_mrk_opts.nRepetitions;

%% Calibration **practice**
fprintf('Press <RETURN> to start speller calibration PRACTICE.\n'); pause;
log_filename = [TODAY_DIR speller_calib_prefix 'practice_' VP_CODE '.log']; %#ok<*NASGU>
desired_phrase = phrase_practice;

setup_online_speller
pyff('setdir','');
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start speller calibration.\n'); pause;
log_filename = [TODAY_DIR speller_calib_prefix VP_CODE '.log'];
desired_phrase = phrase;

setup_online_speller 
pyff('setdir', 'basename', speller_calib_prefix);
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');
fprintf('Speller calibration finished.\n')
