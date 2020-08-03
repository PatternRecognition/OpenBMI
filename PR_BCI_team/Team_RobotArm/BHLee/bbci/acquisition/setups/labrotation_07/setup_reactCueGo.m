go_type= 'visual';

N = 200;
opt.filename = 'react_cue_go';
opt.perc_dev = 0.5;
opt.response_markers = {'R  8', 'R 16'};
opt.bv_host = 'localhost';
opt.intertrial = [1000 500];
opt.duration_cue = 200;
opt.duration_intro = 5000;
opt.position = [-1919 0 1920 1181];
opt.cross = 1;
opt.background = 0.6*[1 1 1];
opt.handle_background = stimutil_initFigure(opt);
opt.countdown = 7;
opt.countdown_fontsize = 0.3;
opt.fs = 22050;
opt.bv_host = 'localhost';
opt.msg_intro = 'relax';
opt.msg_fin = 'pause';

opt.lower_delay = 250;
opt.range_delay = [300 900];
opt.upper_delay = 1000;

nogo_color = 0.3*[1 1 1];
go_color = [1 1 1];
opt.arrow_color = nogo_color;
cues = {'left','right'}; 
[H, opt.handle_cross] = stimutil_cueArrows(cues, opt);
opt.nogo_cue_left  = H(1);
opt.nogo_cue_right = H(2);

opt.stim_type= go_type;
if strcmp(go_type,'visual')
  opt.arrow_color = go_color;
  [H, opt.handle_cross] = stimutil_cueArrows(cues, opt);
  opt.go_cue_left  = H(1);
  opt.go_cue_right = H(2);
elseif strcmp(go_type,'auditory')
  opt.go_cue_left  = stimutil_generateTone(200, 'harmonics',5, 'duration',200, 'pan',[0.1 1]);
  opt.go_cue_right = stimutil_generateTone(500, 'harmonics',5, 'duration',200, 'pan',[1 0.1]);
end

fprintf('for testing:\n  stim_reactCueGo(30, opt, ''test'', 1);\n');
fprintf('stim_reactCueGo(N, opt);\n');

