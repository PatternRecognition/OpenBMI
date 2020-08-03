cd([BCI_DIR 'studies/season1']);

opt_seg= struct('session_start_marker', [252 -252], ...
                'session_end_marker', [253 -253], ...
                'pause_start_marker', [249 -249], ...
                'pause_end_marker', [250 -250]);
opt= struct('xTrials',[5 10]);

expbase= generate_experiment_database;
iNoApp= find(cell2mat(apply_cellwise({expbase.appendix}, 'isempty')));
iImag= intersect(strmatch('imag', {expbase.paradigm}), iNoApp);
imagbase= expbase(iImag);
imagbase= bbcifile_joinParadigms(imagbase, 'imag_move', 'imag_lett');

k= 0;
for ee= [32 35 36 38 40],
  file= bbcifile_compose(imagbase(ee));
  sub_dir= strcat(imagbase(ee).subject, '_', imagbase(ee).date);
  [cnt, mrk, mnt]= loadProcessedEEG(file);

  eval(['setup_' sub_dir]);  
  cnt = proc_selectChannels(cnt, csp.clab);
  [b,a] = butter(csp.filtOrder, csp.band/cnt.fs*2);
  cnt = proc_filt(cnt, b, a);
  combs = nchoosek(1:length(mrk.className), 2);
  for cc= 1:size(combs,1),
    mrk_cl= mrk_selectClasses(mrk, combs(cc,:));
    fv= makeEpochs(cnt, mrk_cl, csp.ival);
    fv.proc= ['fv= proc_csp(epo,' int2str(csp.nPat) '); ' ...
              'fv= proc_variance(fv); ' ...
              'fv= proc_logarithm(fv);'];

    %% cross-validation with simple LDA
    opt.out_prefix= sprintf('%s <%s-%s>: ', sub_dir, mrk_cl.className{:});
    [loss, loss_std, out]= xvalidation(fv, 'LDA', opt);
    k= k+1;
    err(k)= loss;
  end
end
