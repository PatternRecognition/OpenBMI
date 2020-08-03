stim= [];
clear cl;
if length(bbci.classes) <3
  cl(1) = bbci.classes(1);
  cl(2) = bbci.classes(2);
  cl(ismember(bbci.classes, 'foot')) = {'down'}
  stim.cue= struct('classes',cl );
  [stim.cue.nEvents]= deal(22);
else
  cl = {'left','right','down'};
  stim.cue= struct('classes',cl );
  [stim.cue.nEvents]= deal(20);
end  
[stim.cue.timing]= deal([2000 4000 2000]);
stim.msg_intro= 'Gleich geht''s los';
stim.msg_rlx = 'Relax';

opt= [];
opt.position= [-1919 0 1920 1181];
opt.cross= 1;

opt.handle_background= stimutil_initFigure(opt);
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

opt.filename= 'imag_arrow';
opt.breaks= [20 15];
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
