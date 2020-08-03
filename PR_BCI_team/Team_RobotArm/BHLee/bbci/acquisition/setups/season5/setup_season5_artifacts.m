%function [default_seq, sounds]= setup_artifact(varargin);
%[seq, wav]= setup_artifact_measurement(<opt>)
%
%Arguments:
% OPT: Struct or property/value of optional properties:


%if nargout==0,
%    error('you need to specify an output argument');
%end

seq= ['P5000 f2F3P3000 F1P1000 f2F4P3000 F1P1000 f2F5P3000 F1P1000 ' ...
              'P2000 ' ...
              'f6F7P2000 R[5] (F8P1500 F7P1000 F9P1500 F7P1000) ' ...
              'P2000 ' ...
              'R[5] (F10P1500 F7P1000 F11P1500 F7P1000) ' ...
              'P2000 ' ...
              'F12P20000 F1P3000 F13P4000 F1P3000 ' ...
              'R[5] (F14P15000 F1P1000 F15P15000 F1P1000) ' ...
              'P2000 ' ...
              'F16P5000 F1P2000 F17P5000 F1P2000 ' ...
              'F18P5000 F1P2000 F19P5000 F1P2000 F20P1000'];

SOUND_DIR= [BCI_DIR 'data/sound/'];

%opt= propertylist2struct(varargin{:});
%opt= set_defaults(opt, ...
%                  'language', 'english');
opt.language= 'english';
SPEECH_DIR= [SOUND_DIR 'english/'];

switch(lower(opt.language)),
 case {'german','deutsch'}
  error('not implemented')
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
opt.filename= 'arte7';
opt.position= [-1919 0 1920 1181];

fprintf('for testing:\n  stim_artifactMeasurement(seq, wav, opt, ''test'', 1);\n');
fprintf('stim_artifactMeasurement(seq, wav, opt);\n');
