
function online_test_artifacts(wld, file)

% load and filter
Wps = wld.filt_raw/wld.fs_orig*2;
[n, Ws] = cheb2ord(Wps(1), Wps(2), 3, 40);
[filt.b, filt.a] = cheby2(n, 50, Ws);
[cnt, mrk] = eegfile_readBV(file, 'fs', wld.fs, 'filt', filt, 'clab', wld.clab);
    
% block structure    
blk1 = blk_segmentsFromMarkers(mrk,'start_marker',wld.mrk.start_low,'end_marker',wld.mrk.end_low);
blk2 = blk_segmentsFromMarkers(mrk,'start_marker',wld.mrk.start_high,'end_marker',wld.mrk.end_high);
blk  = blk_merge(blk1, blk2, 'className',{'low workload','high workload'});
mkk  = mrk_evenlyInBlocks(blk, wld.T_epo);
    
% check artifacts
reject_varEventsAndChannels(cnt, mkk, [0 wld.T_epo-1],'visualize', 1);

