
%% define parameters:
% trials:
nTrials = 100;
frac_longest = 0.5;
frac_shortest = 0.2;
frac_correct = 0.8;

% times (all in milliseconds):
countdown_from = 4000;
fixation_duration = 1000;
cue_duration = 100;
directingattention_duration = [500, 2000];
target_duration = 200;
masker_duration = 140;
responsetime_duration = 700;
rest_duration = 1000;

% colors:
cue_colors = {[0.0, 1.0, 0.8] ...
              [0.2, 0.0, 0.8] ...
              [0.4, 1.0, 0.8] ...
              [0.2, 0.0, 0.8] ...
              [0.7, 1.0, 0.8] ...
              [0.2, 0.0, 0.8]};
user_cuecolor = 4;

display_radius = 400;
circle_radius = 80;
circle_textsize = 70;
circle_textwidth = 9;
canvas_width = 900;
canvas_height = 900;

%% start feedback:
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CovertAttentionHex', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 'i:nTrials', nTrials);
send_xmlcmd_udp('interaction-signal', 'frac_longest', frac_longest);
send_xmlcmd_udp('interaction-signal', 'frac_shortest', frac_shortest);
send_xmlcmd_udp('interaction-signal', 'frac_correct', frac_correct);
send_xmlcmd_udp('interaction-signal', 'i:countdown_from', countdown_from);
send_xmlcmd_udp('interaction-signal', 'i:fixation_duration', fixation_duration);
send_xmlcmd_udp('interaction-signal', 'i:cue_duration', cue_duration);
send_xmlcmd_udp('interaction-signal', 'i:directingattention_duration', directingattention_duration);
send_xmlcmd_udp('interaction-signal', 'i:target_duration', target_duration);
send_xmlcmd_udp('interaction-signal', 'i:masker_duration', masker_duration);
send_xmlcmd_udp('interaction-signal', 'i:responsetime_duration', responsetime_duration);
send_xmlcmd_udp('interaction-signal', 'i:rest_duration', rest_duration);
send_xmlcmd_udp('interaction-signal', 'cue_colors', cue_colors);
send_xmlcmd_udp('interaction-signal', 'i:user_cuecolor', user_cuecolor);

% Graphical settings
send_xmlcmd_udp('interaction-signal', 'i:display_radius',display_radius);
send_xmlcmd_udp('interaction-signal', 'i:circle_radius',circle_radius);
send_xmlcmd_udp('interaction-signal', 'i:circle_textsize',circle_textsize);
send_xmlcmd_udp('interaction-signal', 'i:circle_textwidth',circle_textwidth);
send_xmlcmd_udp('interaction-signal', 'i:canvas_width',canvas_width);
send_xmlcmd_udp('interaction-signal', 'i:canvas_height',canvas_height);
send_xmlcmd_udp('interaction-signal', 'i:screen_pos',VP_SCREEN);
