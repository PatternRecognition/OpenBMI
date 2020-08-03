N= 100;
calib_set= [14 20 23 26;
            16 21 24 28;
            18 22 25 30];

opt= struct('perc_dev', 28/100);
opt.avoid_dev_repetitions= 1;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 1000;
opt.isi_jitter= 250;
opt.filename= sprintf('mmn_calib_set%d', set_no);;

opt.cue_dev= strcat('mnru', cprintf('%02d', calib_set(set_no,:))', '_a');
opt.cue_std= {'mnru100_a'};

%fprintf('for testing:\n  stim_oddballAuditory(N, opt, ''test'',1);\n');
%fprintf('stim_oddballAuditory(N, opt);\n');
