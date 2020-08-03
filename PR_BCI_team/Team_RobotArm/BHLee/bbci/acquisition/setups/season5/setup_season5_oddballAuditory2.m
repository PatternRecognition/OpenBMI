N= 200;

opt= struct('perc_dev', 0.15);
opt.response_markers= {'R 16', 'R  8'};
opt.bv_host= 'localhost';
opt.position= [-1919 0 1920 1181];
opt.isi= 1600;

opt.cue_std= stimutil_generateTone(200, 'harmonics',5, 'duration',200, 'pan',[0.1 1]);
opt.cue_dev= stimutil_generateTone(500, 'harmonics',5, 'duration',200, 'pan',[1 0.1]);

fprintf('for testing:\n  stim_oddballAuditory(N, opt, ''test'',1);\n');
fprintf('stim_oddballAuditory(N, opt);\n');
