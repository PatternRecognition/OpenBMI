default_crit= struct('maxmin', 100, ...
                     'clab', 'Fp*', ...
                     'ival', [100 800]);
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'reject_eyemovements', 1, ...
                 'reject_eyemovements_crit', default_crit);

[mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 1000], 'visualize', 0, 'do_multipass', 1, 'do_bandpass', 0);

%do the hp filt
hpWps = [0.1, 0.5]/Cnt.fs*2;
[hpn, hpWn] = buttord(hpWps(1), hpWps(2), 3, 50);
[hpfilt.b, hpfilt.a]= butter(hpn, hpWn, 'high');
Cnt = proc_filt(Cnt, hpfilt.b, hpfilt.a);
% end hp filt

epo = cntToEpo(Cnt,mrk_rej,opt.ival);

if opt.reject_eyemovements & opt.reject_eyemovements_crit.maxmin>0,
  epo_crit= proc_selectIval(epo, opt.reject_eyemovements_crit.ival);
  iArte= find_artifacts(epo_crit, opt.reject_eyemovements_crit.clab, ...
                        opt.reject_eyemovements_crit);
  fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
          length(iArte), opt.reject_eyemovements_crit.maxmin);
  clear epo_crit
  epo= proc_selectEpochs(epo, 'not',iArte);
else
  iArte = [];
end


if do_reject_channels,
    epo = proc_selectChannels(epo, 'not', rclab);
else
    rclab = {};
end
% epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
% opt.selectival = select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',3, 'visualize', 0, 'sort', 1);

epo_sampling_ivals = [80,350; 360, 800];
epo_sampling_deltas = [20, 60];
opt.selectival = [];
for k=1:size(epo_sampling_ivals,1)
    t_lims = epo_sampling_ivals(k,:);
    dt = epo_sampling_deltas(k);
    sampling_points = t_lims(1):dt:t_lims(2);
    tmp_ivals = round([sampling_points' - dt/2, sampling_points' + dt/2]);
    opt.selectival = [opt.selectival; tmp_ivals];
end
