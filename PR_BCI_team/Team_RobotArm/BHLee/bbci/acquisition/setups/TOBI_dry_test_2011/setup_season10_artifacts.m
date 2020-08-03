function [seq, wav, opt]= setup_season10_artifacts(varargin)

global BCI_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clstag', 'LRF', ...
                  'language', 'german');

seq= ['P5000 R[2] ('];
if ismember('L', opt.clstag),
  seq= [seq, 'f2F3P3000 F1P1000 '];
end
if ismember('R', opt.clstag),
  seq= [seq, 'f2F4P3000 F1P1000 '];
end
if ismember('F', opt.clstag),
  seq= [seq, 'f2F5P3000 F1P1000 '];
end
seq= [seq, [') P2000 ' ...
      'f6F7P2000 ' ...
      'F8P1500 F7P1000 F9P1500 F7P1000 F10P1500 F7P1000 F8P1500 F7P1000 ' ...
      'F11P1500 F7P1000 F9P1500 F7P1000 F10P1500 F7P1000 F11P1500 F7P1000 ' ...
      'F10P1500 F7P1000 F9P1500 F7P1000 F11P1500 F7P1000 F8P1500 F7P1000 ' ...
      'F9P1500 F7P1000 F8P1500 F7P1000 F11P1500 F7P1000 F10P1500 F7P1000 ' ...
      'F9P1500 F7P1000 F11P1500 F7P1000 F10P1500 F7P1000 F9P1500 F7P1000 ' ...
      'F8P1500 F7P1000 F10P1500 F7P1000 F8P1500 F7P1000 F11P1500 F7P1000 ' ...
      'P2000 ' ...
      'F12P10000 F1P3000 F20P1000']];

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'language', 'german');

if isequal(opt.language,'deutsch')
  opt.language = 'german';
end
SOUND_DIR= [BCI_DIR 'acquisition/data/sound/'];
SPEECH_DIR= [SOUND_DIR opt.language '/'];

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
 otherwise,
  error('unknown language');
end

for i = 1:length(cel)
  [wav(i).sound,wav(i).fs] = wavread([SPEECH_DIR '/speech_' cel{i} '.wav']);
end

opt= [];
opt.handle_background= stimutil_initFigure;
opt.filename= 'artifacts';

desc= stimutil_readDescription('season10_artifacts');
h_desc= stimutil_showDescription(desc, 'waitfor',0);
opt.delete_obj= h_desc.axis;

assignin('base', 'seq',seq);
assignin('base', 'wav',wav);
assignin('base', 'opt',opt);

fprintf('stim_artifactMeasurement(seq, wav, opt);\n');
