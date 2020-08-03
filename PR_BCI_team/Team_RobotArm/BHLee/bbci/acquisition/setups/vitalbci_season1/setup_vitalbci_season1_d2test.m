N= 150;

opt= struct('perc_dev', 0.2);
opt.bv_host= 'localhost';
opt.duration_cue= 250;
opt.response_delay= [500 1000];

opt.duration_response= 1000;
opt.duration_blank= 500;

fprintf('for testing:\n  stim_d2test(N, opt, ''test'',1);\n');
fprintf('stim_d2test(N, opt);\n');
