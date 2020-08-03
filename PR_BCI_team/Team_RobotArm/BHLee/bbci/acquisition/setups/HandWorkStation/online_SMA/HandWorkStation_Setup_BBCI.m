
function bbci = HandWorkStation_Setup_BBCI(wld)


n_bands = size(wld.bands,1);

bbci = struct;
bbci.source.min_blocklength = wld.T_epo;
bbci.source.marker_mapping_fcn = '';


Wps = wld.filt_raw/wld.fs_orig*2;
[n, Ws] = cheb2ord(Wps(1), Wps(2), 3, 40);
[filt_b, filt_a] = cheby2(n, 50, Ws);
bbci.source.acquire_fcn = @bbci_acquire_bv;
bbci.source.acquire_param = struct('fs',wld.fs,'filt_b',filt_b,'filt_a',filt_a);


for ii = 1:n_bands
    bbci.cont_proc(ii).clab = wld.clab;
    bbci.cont_proc(ii).source = 1;
    [filt_low_b, filt_low_a] = butter(6, wld.bands(ii,2)/wld.fs*2, 'low');
    [filt_high_b, filt_high_a] = butter(6, wld.bands(ii,1)/wld.fs*2, 'high');
    bbci.cont_proc(ii).proc = {{@online_filt, filt_low_b, filt_low_a}, ...
                               {@online_filt, filt_high_b, filt_high_a}};
end


for ii = 1:n_bands
    bbci.feature(ii).cont_proc = ii;
    bbci.feature(ii).proc = {{@proc_linearDerivation, wld.W{ii}},...
                              @proc_variance, @proc_logarithm};
    bbci.feature(ii).ival= [-wld.T_epo 0];
end


bbci.classifier.feature= 1:n_bands;
bbci.classifier.C = wld.C;


opt.wld = wld;
bbci.control(1).fcn = @bbci_control_HandWorkStation;
bbci.control(1).param = {opt};

bbci.control(2).fcn = @bbci_control_HandWorkStation;
bbci.control(2).param = {opt};
bbci.control(2).condition.marker = [wld.mrk.start_low...
                                    wld.mrk.start_high...
                                    wld.mrk.force_down...
                                    wld.mrk.force_up];

bbci.feedback.control= 1;
bbci.feedback.receiver= 'pyff';
bbci.feedback(2).control= 2;
bbci.feedback(2).receiver= 'pyff';

                                
bbci.quit_condition.marker = 255;
bbci.quit_condition.running_time = wld.T+15;


bbci.log.output = 'screen';
bbci.log.filebase = '~/bbci/log/log';
bbci.log.classifier = 1;
