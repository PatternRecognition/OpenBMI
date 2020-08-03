%% global variables need to be defined here, since this script might be
%% executed within the run_acquisition_script function.
global BCI_DIR

seq= ['P5000 R[2] (F14P5000 F1P2000 F15P5000 F1P2000) F20P1000'];

opt.language = 'english';
% opt = setdefault(opt,'language','german');
if isequal(opt.language,'deutsch')
  opt.language = 'german';
end

SOUND_DIR= [BCI_DIR 'acquisition/data/sound/'];
SPEECH_DIR= [SOUND_DIR opt.language '/'];

switch(lower(opt.language)),
 case 'german'
   cel = {'stopp', ...                       %% 01
         'anspannen', ...        %% 02
         'links', ...                       %% 03
         'rechts', ...                      %% 04
         'fuss', ...                       %% 05
         'schauen', ...                       %% 06
         'mitte', ...                     %% 07
         'links', ...                       %% 08
         'rechts', ...                      %% 09
         'rauf', ...                         %% 10
         'unten', ...                       %% 11
         'blinzeln', ...                      %% 12
         'augen_fest_zu_druecken', ...    %% 13
         'augen_zu', ...                %% 14
         'augen_auf', ...                  %% 15
         'schlucken', ...                    %% 16
         'zunge_gegen_gaumen_druecken', ...
         'schultern_heben', ...             %% 18
         'zaehne_knirschen',  ...              %% 19
         'vorbei'};                          %% 20
 case 'english'
  cel = {'stop', ...                       %% 01
         'maximum_compression', ...        %% 02
         'left', ...                       %% 03
         'right', ...                      %% 04
         'foot', ...                       %% 05
         'look', ...                       %% 06
         'center', ...                     %% 07
         'left', ...                       %% 08
         'right', ...                      %% 09
         'up', ...                         %% 10
         'down', ...                       %% 11
         'blink', ...                      %% 12
         'press_your_eyelids_shut', ...    %% 13
         'eyes_closed', ...                %% 14
         'eyes_open', ...                  %% 15
         'swallow', ...                    %% 16
         'press_tongue_to_the_roof_of_your_mouth', ...
         'lift_shoulders', ...             %% 18
         'clench_teeth',  ...              %% 19
         'over'};                          %% 20
 otherwise,
  error('unknown language');
end

for i = 1:length(cel)
  [wav(i).sound,wav(i).fs] = wavread([SPEECH_DIR '/speech_' cel{i} '.wav']);
end

opt= [];
opt.handle_background= stimutil_initFigure;
opt.filename= 'artifacts';
opt.test= 1;

desc= stimutil_readDescription('adaptationstudy_season1_artifacts');
h_desc= stimutil_showDescription(desc, 'waitfor',0);
opt.delete_obj= h_desc.axis;

fprintf('This is just for testing (not recording EEG):\n');
fprintf('stim_artifactMeasurement(seq, wav, opt, ''test'', 1);\n');
