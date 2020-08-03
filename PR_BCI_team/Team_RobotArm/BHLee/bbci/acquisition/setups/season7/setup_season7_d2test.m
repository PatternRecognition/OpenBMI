N= 150;
opt= struct('perc_dev', 0.2);
opt.bv_host= 'localhost';
opt.position= [-1919 -149 1920 1181];

fprintf('for testing:\n  stim_d2test(N, opt, ''test'',1);\n');

fprintf('stim_d2test(N, opt);\n');
