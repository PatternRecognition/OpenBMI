if ~exist('classes', 'var'),
  error('You must define variable ''classes''.');
end
fprintf('Using classes: %s\n', vec2str(classes));

stim.cue= struct('classes', classes);
[stim.cue.nEvents]= deal(50);
[stim.cue.timing]= deal([2000 4000 2000]);
markers= num2cell(find(ismember({'left','right','foot'}, classes)));
[stim.cue.marker]= deal(markers{:});

opt= [];
opt.position= [-1919 0 1920 1200];

opt.handle_background= stimutil_initFigure(opt);
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

opt.filename= 'imag_arrow';
opt.breaks= [15 15];  %% Alle 15 Stimuli Pause fuer 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
