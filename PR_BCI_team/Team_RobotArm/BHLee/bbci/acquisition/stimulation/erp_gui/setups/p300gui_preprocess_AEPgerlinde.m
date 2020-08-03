% [mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 0, 'do_multipass', 1, 'do_bandpass', 0, 'whiskerlength', 2.5);
[mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 0, 'do_multipass', 1, 'do_bandpass', 0, 'whiskerlength', 2.5);
epo = cntToEpo(Cnt,mrk_rej,opt.ival);
if do_reject_channels,
    epo = proc_selectChannels(epo, 'not', rclab);
else
    rclab = {};
end
epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
opt.selectival = select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',3, 'visualize', 0, 'sort', 1);