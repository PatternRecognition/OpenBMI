function [cnt, mrk, sbjData] = load_data_MVEP(VPDir, varargin)

% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt ...
%     );

sbj = strtok(VPDir, '_');
file_name = fullfile(VPDir, ['covert_cake_' sbj '_100Hz']);
[cnt, mrk]= eegfile_loadMatlab(file_name);
mrk.className = {'Target', 'Non-target'};

% for the free-spelling phase, the marker values must be adjusted to
% reflect target/non-target info
% free_spelling_epos = mrk.mode(3,:)==1;
% y = mrk.y(1,free_spelling_epos);
% toe = mrk.toe(free_spelling_epos);
% toe(y==1) = toe(y==1)+40;
% mrk.toe(free_spelling_epos) = toe;

% load the classifier
s = eegfile_loadMatlab(file_name, 'vars', {'bbci'});
sbjData = [];
sbjData.selectival = s.bbci.analyze.ival;
sbjData.classifier = s.cls;
cnt = proc_selectChannels(cnt, s.bbci.analyze.features.clab);