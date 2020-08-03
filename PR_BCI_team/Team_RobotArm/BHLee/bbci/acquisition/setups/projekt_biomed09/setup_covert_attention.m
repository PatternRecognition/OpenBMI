%% Settings for pyff 

fps = 30;
cue_size = 40;
fixationpoint_radius = 4;
htarget_color = ones(3,1)*157;

nTrials = 100;

width = VP_SCREEN(3);
height = VP_SCREEN(4);
target1center = [width/8 height/8*7];
target2center = [width/8*7 height/8*7];
targetSize = [120 50];
filepath = 'D:\svn\pyff\src\Feedbacks\sound.wav';


send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CovertAttentionFeedback','command','sendinit');  % Choose Feedback
pause(3)
send_xmlcmd_udp('interaction-signal', 'i:screen_pos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:fps',fps);
send_xmlcmd_udp('interaction-signal', 'i:cue_size',cue_size);
send_xmlcmd_udp('interaction-signal', 'i:fixationpoint_radius',fixationpoint_radius);

send_xmlcmd_udp('interaction-signal', 'i:htargetColor',htarget_color);
send_xmlcmd_udp('interaction-signal', 'i:nTrials',nTrials);
send_xmlcmd_udp('interaction-signal', 'i:target1center',target1center);
send_xmlcmd_udp('interaction-signal', 'i:target2center',target2center);
send_xmlcmd_udp('interaction-signal', 'i:targetSize',targetSize);

send_xmlcmd_udp('interaction-signal', 's:soundFilepath',filepath);
