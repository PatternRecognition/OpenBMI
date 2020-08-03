if ~exist('classes', 'var'),
  error('You must define variable ''classes''.');
end
fprintf('Using classes: %s\n', vec2str(classes));

stim.cue= struct('classes', classes);
[stim.cue.nEvents]= deal(30);
[stim.cue.timing]= deal([1500 1500 0; 6500 6500 0]);
markers= num2cell(find(ismember({'left','right','foot'}, classes)));
[stim.cue.marker]= deal(markers{:});

opt= [];
opt.filename= 'imag_audicompetition';
opt.breaks= [30 15];
opt.position= [0 800 800 600];
opt.break_pause_recording= 1;

fprintf('for testing:\n  stim_auditoryCues(stim, opt, ''test'',1);\n');
fprintf('stim_auditoryCues(stim, opt);\n');
