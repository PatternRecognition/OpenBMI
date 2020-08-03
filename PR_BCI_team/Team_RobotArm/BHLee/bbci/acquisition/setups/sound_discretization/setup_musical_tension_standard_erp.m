N= 500;

opt= struct('perc_dev', 15/100);
opt.avoid_dev_repetitions= 1;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 500;
opt.fixation= 1;
opt.filename= 'mmn_auditory';
opt.speech_intro= '';
opt.msg_intro= '';
opt.countdown=5;

semi= 2^(1/12);
%c_freq= 440*semi^3;
a_freq= 440;
d_freq= 440*semi^5;

opt.cue_std= stimutil_generateTone(a_freq, 'harmonics',1, ...
  'duration',250, ...
  'pan', 0.25*[1 1], ...
  'rampon',25, ...
  'rampoff',25);

opt.cue_dev= stimutil_generateTone(d_freq, 'harmonics',1, ...
  'duration',250, ...
  'pan', 0.25*[1 1], ...
  'rampon',25, ...
  'rampoff',25);

opt.handle_background= stimutil_initFigure;


desc= stimutil_readDescription('musical_tension_standard_erp');

h_desc= stimutil_showDescription(desc, 'waitfor',4);
%opt.delete_obj= h_desc.axis;

%fprintf('stim_artifactMeasurement(seq, wav, opt);\n');

fprintf('for testing:\n  stim_oddballAuditory(N, opt, ''test'',1);\n');
fprintf('stim_oddballAuditory(N, opt);\n');
