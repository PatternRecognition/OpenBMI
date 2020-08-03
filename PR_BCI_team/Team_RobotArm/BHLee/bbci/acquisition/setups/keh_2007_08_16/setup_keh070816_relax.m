%function [default_seq, sounds]= setup_artifact(varargin);
%[seq, wav]= setup_artifact_measurement(<opt>)
%
%Arguments:
% OPT: Struct or property/value of optional properties:


%if nargout==0,
%    error('you need to specify an output argument');
%end

seq= ['P5000 ' ...
      'R[5] (F14P60000 F1P3000 F15P60000 F1P3000)'];

opt= [];
opt.language= 'german';

%opt= propertylist2struct(varargin{:});
%opt= set_defaults(opt, ...
%                  'language', 'english');
SPEECH_DIR= [BCI_DIR 'data/sound/'];

switch(lower(opt.language)),
 case {'german','deutsch'}
  cel = {'stopp', ...                       %% 01
          'xx_maximum_compression', ...        %% 02
          'links', ...                       %% 03
          'rechts', ...                      %% 04
          'Fuss', ...                       %% 05
          'xx_look', ...                       %% 06
          'mitte', ...                     %% 07
          'links', ...                       %% 08
          'rechts', ...                      %% 09
          'oben', ...                         %% 10
          'unten', ...                       %% 11
          'blinzeln', ...                      %% 12
          'xx', ...    %% 13
          'Augen_zu', ...                %% 14
          'Augen_offen', ...                  %% 15
          'xx_schlucken', ...                    %% 16
          'xx_press_tongue_to_the_roof_of_your_mouth', ...
          'xx_lift_shoulders', ...             %% 18
          'Zaehne_knirschen',  ...              %% 19
          'xx_over'};                          %% 20
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
  if ~strcmp(cel{i}(1:2),'xx'),
    [wav(i).sound,wav(i).fs] = wavread([SPEECH_DIR '/' cel{i} '.wav']);
  end
end

opt= [];
opt.filename= 'arte7';
fprintf('for testing:\n  stim_artifactMeasurement(seq, wav, opt, ''test'', 1);\n');
fprintf('stim_artifactMeasurement(seq, wav, opt);\n');
