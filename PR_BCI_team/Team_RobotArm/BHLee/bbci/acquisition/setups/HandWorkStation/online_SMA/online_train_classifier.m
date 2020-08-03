
function wld = online_train_classifier(wld, files)

nBands = size(wld.bands,1);
fv = cell(1,nBands);

for bb = 1:length(files)
    
    % load and filter
    Wps = wld.filt_raw/wld.fs_orig*2;
    [n, Ws] = cheb2ord(Wps(1), Wps(2), 3, 40);
    [filt.b, filt.a] = cheby2(n, 50, Ws);
    [cnt, mrk] = eegfile_readBV(files{bb}, 'fs', wld.fs, 'filt', filt, 'clab', wld.clab);
    
    % block structure
    blk1 = blk_segmentsFromMarkers(mrk,'start_marker',wld.mrk.start_low,'end_marker',wld.mrk.end_low);
    blk2 = blk_segmentsFromMarkers(mrk,'start_marker',wld.mrk.start_high,'end_marker',wld.mrk.end_high);
    blk1.ival(1,:) = blk1.ival(1,:) + wld.block_offset*blk1.fs;
    blk2.ival(1,:) = blk2.ival(1,:) + wld.block_offset*blk2.fs;
    blk  = blk_merge(blk1, blk2, 'className',{'low workload','high workload'});
    mkk  = mrk_evenlyInBlocks(blk, wld.T_epo);
    
    % reject epochs with artifacts
    mkk = reject_varEventsAndChannels(cnt, mkk, [0 wld.T_epo-1],'visualize', 0);

    % band pass filtering
    for ff = 1:nBands
        [b,a] = butter(6, wld.bands(ff,1)/cnt.fs*2, 'high');
        cnt_flt= proc_channelwise(cnt, 'filt', b, a);
        [b,a] = butter(6, wld.bands(ff,2)/cnt.fs*2, 'low');
        cnt_flt= proc_channelwise(cnt_flt, 'filt', b, a);
        fv{ff} = proc_appendEpochs(fv{ff},cntToEpo(cnt_flt, mkk, [0 wld.T_epo], 'mtsp', 'before'));
    end
    
end

clear cnt cnt_flt

% compute CSPs
for ff = 1:nBands
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


