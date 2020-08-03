N= 150;

opt= struct('perc_dev', 0.5);
opt.response_markers= {'R 16', 'R  8'};
opt.bv_host= 'localhost';
opt.position= [-1919 0 1920 1181];
opt.duration_cue= 250;
opt.response_delay= [700 1700];
opt.duration_response= 1000;
opt.duration_blank= 500;
%opt.position= [5 570 640 480];

fprintf('for testing:\n  stim_d2test(N, opt, ''test'',1);\n');
fprintf('stim_d2test(N, opt);\n');
