%function [default_seq, sounds]= setup_artifact(varargin);
%[seq, wav]= setup_artifact_measurement(<opt>)
%
%Arguments:
% OPT: Struct or property/value of optional properties:


%if nargout==0,
%    error('you need to specify an output argument');
%end

seq= ['P5000 ' ...
      'R[10] (F14P30000 F1P3000 F15P30000 F1P3000) F20'];

opt= [];
opt.language= 'german';

switch(lower(opt.language)),
 case {'german','deutsch'}
  SPEECH_DIR= [SOUND_DIR 'german/'];
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
  SPEECH_DIR= [SOUND_DIR 'english/'];
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
opt.filename= 'artifacts';
fprintf('for testing:\n  stim_artifactMeasurement(seq, wav, opt, ''test'', 1);\n');
fprintf('stim_artifactMeasurement(seq, wav, opt);\n');
