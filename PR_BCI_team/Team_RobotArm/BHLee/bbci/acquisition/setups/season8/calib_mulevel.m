global EEG_RAW_DIR

[cnt mrk_orig]= eegfile_readBV([subdir '/imag_fbarrow*'],'fs',bbci.fs,'filt',bbci.filt,'clab',bbci.clab);
cnt = proc_laplacian(cnt, 'require_complete_neighborhood', 0);
cnt = proc_selectChannels(cnt, {'C3 lap', 'C4 lap'});
cnt = proc_filt(cnt, analyze.csp_b, analyze.csp_a);
mrk = mrkodef_imag_fbarrow(mrk_orig, strukt('classes_fb', bbci.classes));
mrk = mrk_selectClasses(mrk, bbci.classes);
epo = makeEpochs(cnt, mrk, analyze.ival);
epo = proc_variance(epo);
epo = proc_logarithm(epo);
epo = proc_average(epo, 'std', 1);
epo.x = epo.x - 1*epo.std;
mu.min = [epo.x(1, 1, 2) epo.x(1, 2, 1)];

[cnt, mrk_orig]= eegfile_readBV([subdir '/artifact*'],'fs',bbci.fs,'filt',bbci.filt);
cnt = proc_laplacian(cnt, 'require_complete_neighborhood', 0);
cnt = proc_selectChannels(cnt, {'C3 lap', 'C4 lap'});
cnt = proc_filt(cnt, analyze.csp_b, analyze.csp_a);
mrk = mrkodef_artifacts(mrk_orig);
mrk_ = mrk_selectClasses(mrk, {'eyes_open'});
mrk_.pos = reshape((repmat(mrk_.pos', 1, 15) + repmat(mrk_.fs*(1:15), length(mrk_.pos), 1))', 1, []);
mrk_.y = ones(size(mrk_.pos));
epo = makeEpochs(cnt, mrk_, [0 1000]);
epo = proc_variance(epo);
epo = proc_logarithm(epo);
epo = proc_average(epo, 'std', 1);
mu.max = epo.x+1*epo.std;


