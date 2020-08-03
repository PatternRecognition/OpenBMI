function [seq, wav, opt]= setup_artifacts_and_resting_measurement(varargin)

global BCI_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'language', 'german', ...
                  'seq', [], ...
                  'show_description', 1);

if isequal(opt.language,'deutsch')
  opt.language = 'german';
end
SOUND_DIR= [BCI_DIR 'acquisition/data/sound/'];
SPEECH_DIR= [SOUND_DIR opt.language '/'];

seq = opt.seq;

switch(lower(opt.language)),
 case {'german','deutsch'}
  cel = {'stopp', ...                          %% 01
         'anspannen', ...                      %% 02
         'links', ...                          %% 03
         'rechts', ...                         %% 04
         'fuss', ...                           %% 05
         'augen', ...                          %% 06
         'mitte', ...                          %% 07
         'links', ...                          %% 08
         'rechts', ...                         %% 09
         'oben', ...                           %% 10
         'unten', ...                          %% 11
         'blinzeln', ...                       %% 12
         'augen_fest_zu_druecken', ...         %% 13
         'augen_zu', ...                       %% 14
         'augen_auf', ...                      %% 15
         'schlucken', ...                      %% 16
         'zunge_gegen_gaumen_druecken', ...    %% 17
         'schultern_heben', ...                %% 18
         'zaehne_knirschen',  ...              %% 19
         'vorbei', ...                         %% 20
         'entspannen'};                        %% 21
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
         'over', ...                       %% 20
         'relax'};                         %% 21
 case 'italian'
     cel = {'stop', ...                       %% 01
         'maximum_compression', ...        %% 02 % non existent
         'sinistra', ...                   %% 03
         'destra', ...                     %% 04
         'foot', ...                       %% 05 % non existent
         'guardare_a', ...                 %% 06
         'centro', ...                     %% 07
         'sinistra', ...                   %% 08
         'destra', ...                     %% 09
         'su', ...                         %% 10
         'giu', ...                        %% 11
         'battere_gli_occhi', ...          %% 12
         'press_your_eyelids_shut', ...    %% 13 % non existent
         'occhi_chiusi', ...               %% 14
         'occhi_aperti', ...               %% 15
         'swallow', ...                    %% 16 % non existent
         'press_tongue_to_the_roof_of_your_mouth', ... %% 17 % non existent
         'lift_shoulders', ...             %% 18 % non existent
         'clench_teeth',  ...              %% 19 % non existent
         'fine', ...                       %% 20
         'relax'};                         %% 21
    otherwise,
        error('unknown language');
end

for i = 1:length(cel)
  [wav(i).sound,wav(i).fs] = wavread([SPEECH_DIR '/speech_' cel{i} '.wav']);
end

% opt= [];
if opt.show_description
    desc= stimutil_readDescription('season10_artifacts');
    h_desc= stimutil_showDescription(desc, 'waitfor',0);
    opt.delete_obj= h_desc.axis;
else
    opt.delete_obj = [];
end

figure();
opt.handle_background= stimutil_initFigure;
opt.filename= 'artifacts';
% assignin('base', 'seq',seq);
% assignin('base', 'wav',wav);
% assignin('base', 'opt',opt);
