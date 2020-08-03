%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause

%% Vormessungen
% Run artifact measurement, oddball practice and oddball test
% vormessungen

%% Basic settings
COPYSPELLING_FINISHED = 246;
phrase_practice = 'BBCI';
% phrases = {'THERE_WAS_A_TABLE_SET_OUT_UNDER_A_TREE_IN_FRONT_OF_THE_HOUSE,_AND_THE_MARCH_HARE_AND_THE_HATTER_WERE_HAVING_TEA_AT_IT._A_DORMOUSE_WAS_SITTING_BETWEEN_THEM,_FAST_ASLEEP.', ...
%           'HAVE_SOME_WINE,_THE_MARCH_HARE_SAID_IN_AN_ENCOURAGING_TONE._ALICE_LOOKED_ALL_ROUND_THE_TABLE,_BUT_THERE_WAS_NOTHING_ON_IT_BUT_TEA._I_DONT_SEE_ANY_WINE,_SHE_REMARKED.'};
% element_count (without errors & correction): [133, 123, 107, 122, 107, 74]
phrases = {'THE_MARCH_HARE_AND_THE_HATTER_WERE_HAVING_TEA._A_DORMOUSE_WAS_SITTING_BETWEEN_THEM.', ...
           'NO_ROOM._THEY_CRIED_OUT_WHEN_THEY_SAW_ALICE_COMING._THERES_PLENTY_OF_ROOM._SAID_ALICE_INDIGNANTLY.', ...
           'HAVE_SOME_WINE,_THE_MARCH_HARE_SAID_IN_AN_ENCOURAGING_TONE.', ...
           'ALICE_LOOKED_ALL_ROUND_THE_TABLE,_BUT_THERE_WAS_NOTHING_ON_IT_BUT_TEA.'};
% element_count (without errors & correction): [112, 118, 120, 106, 101, 63]
         
% Python code for counting appearance of the six elements:
% s = '' # put the whole string in here.
% s = s.upper()
% s = s.replace(' ', '_')
% letter_set = [['A','B','C','D','E'], \
%               ['F','G','H','I','J'], \
%               ['K','L','M','N','O'], \
%               ['P','Q','R','S','T'], \
%               ['U','V','W','X','Y'], \
%               ['Z','_','.',',','<']]
% letter_count = [[],[],[],[],[],[]]
% for i in range(6):
%     for j in range(5):
%         letter_count[i].append(s.count(letter_set[i][j]))
% letter_count = NP.concatenate((NP.array(letter_count).T, NP.array([[0,0,0,0,0,0]]))).T
% element_count = [0,0,0,0,0,0]
% for i in range(6):
%     element_count[i] = sum(letter_count[i,:]) + sum(letter_count[:,i])

l = [length(phrases{1}), length(phrases{2}), length(phrases{3}), length(phrases{4})];
sum(l)
nBlocks = length(phrases);
         
%% Start pyff & BV-Recorder
pyff('startup', 'a', [BCI_DIR 'python/pyff/src/Feedbacks']);
bvr_sendcommand('viewsignals'); pause(5);

%% Practice...
fprintf('Press <RETURN> to start practice.\n'),pause

setup_ErrP_calibration
pyff('set', 'desired_phrase', phrase_practice);
pyff('set', 'log_filename', [TODAY_DIR 'ErrP_calibration_practice_' VP_CODE '.log']);
pyff('setdir','');

fprintf('Ok, starting...\n');
pyff('play');
stimutil_waitForMarker({'R  1', ['S' num2str(COPYSPELLING_FINISHED)]});
fprintf('Block finished.\n')
pyff('stop');
pyff('quit');

%% Copy-spelling Experiment
for iii=1:nBlocks

  fprintf('Press <RETURN> to proceed with block %d\n',iii),pause

  setup_ErrP_calibration
  pyff('set', 'desired_phrase', phrases{iii});
  if iii==1, pyff('set', 'log_filename', [TODAY_DIR 'ErrP_calibration_' VP_CODE '.log']);
  else       pyff('set', 'log_filename', [TODAY_DIR 'ErrP_calibration_' VP_CODE '0' num2str(iii) '.log']); end
  pyff('setdir','basename', 'ErrP_calibration_');

  fprintf('Ok, starting...\n');
  pyff('play');
  stimutil_waitForMarker({'R  1', ['S' num2str(COPYSPELLING_FINISHED)]});
  fprintf('Block finished.\n')

  pyff('stop');
  pyff('quit');


end

fprintf('Experiment finished.\n');
