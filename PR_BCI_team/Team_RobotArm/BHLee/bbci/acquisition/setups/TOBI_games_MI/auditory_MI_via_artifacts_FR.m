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
      'R[3](F2P6000 F4P7000' ...
      'F2P6000 F4P7000' ...
      'F2P6000 F4P7000' ...
      'F3P6000 F4P7000' ...
      'F2P6000 F4P7000' ...
      'F3P6000 F4P7000' ...
      'F3P6000 F4P7000' ...
      'F2P6000 F4P7000' ...
      'F3P6000 F4P7000)' ...
      'F5P6000' ...   
     ]];

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
  cel = {'links', ...                          %% 01
         'rechts', ...                         %% 02
         'fuss', ...                           %% 03
         'stopp', ...                          %% 04
         'vorbei'};                            %% 05
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
