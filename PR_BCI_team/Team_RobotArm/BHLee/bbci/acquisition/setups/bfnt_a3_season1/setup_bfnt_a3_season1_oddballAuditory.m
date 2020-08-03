N= 300;

opt= struct('perc_dev', 30/100);
opt.avoid_dev_repetitions= 1;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 1000;
opt.isi_jitter= 250;

%opt.cue_dev= strcat('mnru', {'00','17','20','25'}, '_jan01');
%opt.cue_std= {'mnru100_jan01'};

devs= strcat('mnru', cprintf('%02d', selected_set)', '_a');
opt.cue_dev= cat(2, {'mnru05_a'}, devs, {'mnru100_i'});
opt.cue_std= {'mnru100_a'};

%fprintf('for testing:\n  stim_oddballAuditory(N, opt, ''test'',1);\n');
%fprintf('stim_oddballAuditory(N, opt);\n');
