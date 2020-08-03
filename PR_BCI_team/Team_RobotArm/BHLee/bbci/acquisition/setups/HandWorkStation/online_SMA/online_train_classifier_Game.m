
function wld = online_train_classifier_Game(wld, files)

nBands = size(wld.bands,1);
fv = cell(1,nBands);

for bb = 1:length(files)
    
    % load and filter
    Wps = wld.filt_raw/wld.fs_orig*2;
    [n, Ws] = cheb2ord(Wps(1), Wps(2), 3, 40);
    [filt.b, filt.a] = cheby2(n, 50, Ws);
    [cnt, mrk] = eegfile_readBV(files{bb}, 'fs', wld.fs, 'filt', filt, 'clab', wld.clab);
        
    
    N_rec = 120*wld.fs;
    N_start = 10*wld.fs;
    N_stop  = 30*wld.fs;
    N_blk = 30*wld.fs;
    rec_offset = cnt.T - N_rec;
    blk.fs = cnt.fs;
    blk.className = {'low workload','high workload'};
    blk.y = [repmat([1 0],1,2) ; repmat([0 1],1,2)];
    blk.ival = [N_blk*[0:3] + rec_offset + N_start; ...
                N_blk*[0:3] + rec_offset + N_stop];
    mkk = mrk_evenlyInBlocks(blk, wld.T_epo);
    
    
    mkk = reject_varEventsAndChannels(cnt, mkk, [0 999],'visualize', 0);

    
    % make epochs
    for ff = 1:nBands
        % band pass filter
        [b,a] = butter(6, wld.bands(ff,1)/cnt.fs*2, 'high');
%        cnt_flt = proc_filt(cnt, b, a);
        cnt_flt= proc_channelwise(cnt, 'filt', b, a);
        [b,a] = butter(6, wld.bands(ff,2)/cnt.fs*2, 'low');
%        cnt_flt = proc_filt(cnt_flt, b, a);
        cnt_flt= proc_channelwise(cnt_flt, 'filt', b, a);
        % epoch data
        fv{ff} = proc_appendEpochs(fv{ff},cntToEpo(cnt_flt, mkk, [0 wld.T_epo], 'mtsp', 'before'));
    end
end
clear cnt cnt_flt

% compute CSSPs
for ff = 1:nBands
    %fv{ff}             = proc_addDelayedChannels(fv{ff}, wld.tau);
    [fv{ff} wld.W{ff}] = proc_csp_auto(fv{ff});
    fv{ff}             = proc_variance(fv{ff});
    fv{ff}             = proc_logarithm(fv{ff});
end
fv2 = fv{1};
for ff = 2:nBands
    fv2 = proc_catFeatures(fv2,fv{ff});
end

% train classifier
wld.C = trainClassifier(fv2, 'RLDAshrink');


