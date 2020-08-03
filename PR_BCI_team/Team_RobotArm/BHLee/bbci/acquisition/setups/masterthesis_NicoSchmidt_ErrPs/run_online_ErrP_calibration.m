%% Basic settings
phrase_practice = 'BCI';
phrase = {'THE_MARCH_HARE_AND_THE_HATTER_WERE_HAVING_TEA._A_DORMOUSE_WAS_SITTING_BETWEEN_THEM.', ...
          'NO_ROOM._THEY_CRIED_OUT_WHEN_THEY_SAW_ALICE_COMING._THERES_PLENTY_OF_ROOM._SAID_ALICE_INDIGNANTLY.', ...
          'HAVE_SOME_WINE,_THE_MARCH_HARE_SAID_IN_AN_ENCOURAGING_TONE.', ...
          'ALICE_LOOKED_ALL_ROUND_THE_TABLE,_BUT_THERE_WAS_NOTHING_ON_IT_BUT_TEA.'};%, ...
%           'HAVE_SOME_WINE,THE_MARCH_HARE_SAID._ALICE_LOOKED_ALL_ROUND_THE_TABLE,BUT_THERE_WAS_NOTHING_ON_IT_BUT_TEA.'
phrase = phrase{1};

%% Calibration **practice**
fprintf('Press <RETURN> to start ErrP calibration practice.\n'),pause
log_filename = [TODAY_DIR ErrP_calib_prefix 'practice_' VP_CODE '.log']; %#ok<*NASGU>
desired_phrase = phrase_practice;

setup_online_ErrP_calibration
pyff('setdir','');
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start ErrP calibration\n'),pause
log_filename = [TODAY_DIR ErrP_calib_prefix VP_CODE '.log']; %#ok<*NASGU>
desired_phrase = phrase;

setup_online_ErrP_calibration
pyff('setdir', 'basename', ErrP_calib_prefix);
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');
fprintf('ErrP calibration finished.\n')
