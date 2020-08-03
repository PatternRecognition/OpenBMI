stim= [];
stim.cue= struct('classes', {'left','right'});
% number of trials of each class 35:
[stim.cue.nEvents]= deal(35);
% Timing: 1. number: fixation cross
%         2. number: duration of cue
%         3. number: duration of blank screen
[stim.cue.timing]= deal([1000 3000 2000]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

% Instruction are in the folder:
%   C:\svn\bbci\acquisition\data\task_descriptions
desc= stimutil_readDescription('vitalbci_season1_imag_arrow');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'imag_arrow';
opt.breaks= [25 15];  %% Alle 25 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;
