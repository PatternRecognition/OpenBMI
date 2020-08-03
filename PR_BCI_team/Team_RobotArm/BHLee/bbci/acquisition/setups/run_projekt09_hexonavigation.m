% if you want to execute only some trials, maybe because the experiment
% broke down before, set e.g. which_trials = 5:7
% trials that don't exist are ignored, so it is safe to write
% which_trials = 8:999

fprintf('Start the feedback controller and press <RET> to continue.\n');
pause;
%system('cmd /C "c: & cd \svn\pyff\src & python FeedbackController.py -p FeedbackControllerPlugins  --additional-feedback-path=C:\svn\bbci\python\pyff\src\Feedbacks" &')
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

fprintf('TEST RUN! Put the screen in horizontal position and press <RET> to continue.\n');
pause;

send_xmlcmd_udp('fc-signal', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'HexoNavi');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
send_xmlcmd_udp('interaction-signal', 'use_horizontal_tactors', 1);
send_xmlcmd_udp('interaction-signal', 'return_to_start', 1);
send_xmlcmd_udp('interaction-signal', 'do_visual_stimulation', 1);
send_xmlcmd_udp('interaction-signal', 'do_tactile_stimulation', 1);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
input('Give focus to Matlab terminal and press <RET> to end the training.\n');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
 

%% build up trials structure
trials = [];
rand('seed',sum(TODAY_DIR));

%first six
condition_tag= {'tactile','visual','bimodal'};
for rr= 1:2,
    idx_block = randperm(length(condition_tag));
    for bb= 1:length(condition_tag)
	this_trial = [];
        this_trial.condition = condition_tag{idx_block(bb)};
        this_trial.do_visual = ismember(this_trial.condition, {'visual','bimodal'});
        this_trial.do_tactile = ismember(this_trial.condition, {'tactile','bimodal'}); 
        this_trial.soa = 8;
        this_trial.sdur = 5;
        this_trial.horiz = true;
        this_trial.level = ['level_balanced' int2str(rr)+1 '.txt'];
        this_trial.msg_after = [];
        trials = [trials; this_trial]; 
    end
end
trials(end).msg_after = 'Put the screen in upright position and press "go<RET>" to continue.';

%with vertical screen
condition_tag= {'tactile_tvsv','tactile_thsv'};
for rr= 1:2,
  idx_block= randperm(length(condition_tag));
  for bb= 1:length(condition_tag)
        this_trial = [];
        this_trial.condition = condition_tag{idx_block(bb)};
        this_trial.do_visual = false;
        this_trial.do_tactile = true;
        this_trial.soa = 8;
        this_trial.sdur = 5;
        %this_trial.horiz = logical(strpatternmatch('*_th*', this_trial.condition));
        if strpatternmatch('*_th*', this_trial.condition)
            this_trial.horiz = true;
        else
            this_trial.horiz = false;
        end
        this_trial.level = ['level_balanced' int2str(rr)+1 '.txt'];
        this_trial.msg_after = [];
        trials = [trials; this_trial]; 
  end
end
trials(end).msg_after = 'Lay the screen down again and press "go<RET>" to continue.';

%slow tactile
for rr= 1:2,
    this_trial = [];
    for bb= 1:length(condition_tag)
        this_trial = [];
        this_trial.condition = 'tactile_slow';
        this_trial.do_visual = false;
        this_trial.do_tactile = true; 
        this_trial.soa = 12;
        this_trial.sdur = 5;
        this_trial.horiz = true;
        this_trial.level = ['level_balanced' int2str(rr)+1 '.txt'];
        this_trial.msg_after = [];
        trials = [trials; this_trial]; 
    end
end
trials(end).msg_after = 'We are done! Press "go<RET>" to continue.';


%% run experiment

fprintf('Going to real recording now.\n');

if ~exist('which_trials','var') || isempty(which_trials)
    which_trials = 1:numel(trials);
end

%ignore trials that don't exist
which_trials = which_trials(which_trials<=numel(trials));

for i=which_trials
    fprintf('Run %d (%s)\n', i, trials(i).condition);
    fprintf('Give focus to Hex window and tell subject to press <SPACE>.\n');
    fbname= ['hexonavi_' trials(i).condition '_P300'];
    send_xmlcmd_udp('fc-signal', 's:TODAY_DIR', TODAY_DIR, 's:VP_CODE', VP_CODE, 's:BASENAME', fbname);
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'HexoNavi');
    send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
    pause(0.5);
    send_xmlcmd_udp('interaction-signal', 'do_visual_stimulation', trials(i).do_visual);
    send_xmlcmd_udp('interaction-signal', 'do_tactile_stimulation', trials(i).do_tactile);
    send_xmlcmd_udp('interaction-signal', 'i:stimulus_onset_asynchrony_in_frames', trials(i).soa);
    send_xmlcmd_udp('interaction-signal', 'i:stimulus_duration_in_frames', trials(i).sdur);
    send_xmlcmd_udp('interaction-signal', 'use_horizontal_tactors',trials(i).horiz);
    send_xmlcmd_udp('interaction-signal', 'return_to_start', 1);
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    pause(30);
    stimutil_waitForInput('phrase', 'go', ...
       'msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
    if ~isempty(trials(i).msg_after)
        stimutil_waitForInput('phrase', 'go', 'msg', trials(i).msg_after);
    end
end
