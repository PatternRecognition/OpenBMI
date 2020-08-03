export_dir= [DATA_DIR 'eegExport/siamac/'];

clab={'F5', 'F3', 'F1', 'Fz', 'F2', 'F6', ...
      'FC5', 'FC3', 'FC1', 'FCz', 'FC2', ...
      'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', ...
      'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', ...
      'CP5', 'CP3', 'CP1', 'CPz', 'CP2', ...
      'CP4', 'CP6', 'TP8', 'P5', 'P3', 'P1', ...
      'Pz', 'P2', 'P4', 'P6', 'P8', 'PO5', 'PO3', 'PO1', ...
      'POz', 'PO2', 'PO4', 'PO6', 'O1', 'Oz', 'O2'};

classes= {'left','right'};
dd= [BCI_DIR 'investigation/studies/'];

list= {'season5/session_list', ...
       'season5/session_list', ...
       'season5/session_list', ...
       'season5/session_list_part2', ...
       'season5/session_list_part2', ...
       'season5/session_list_part2', ...
       'season7/session_list', ...
       'season7/session_list', ...
       'season7/session_list', ...
       'vitalbci_season1/session_list_pilot', ...
       'vitalbci_season1/session_list_tuebingen_pilot'};
filetype= {'imag_audi',
           'imag_lett',
           'imag_move',
           'imag_audi',
           'imag_arrow',
           'imag_move',
           'imagwp1_lett',
           'imagwp2_lett',
           'imagwp3_lett',
           'imag_arrow', 
           'imag_arrow'};           

opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
               'std_of_means',0);

k= 0;
for li= 1:length(list),
  subdir_list= textread([dd list{li}], '%s');
  file= filetype{li};
  for lj= 1:length(subdir_list),
    subdir= subdir_list{lj};
    is= min(find(subdir=='_'));
    sbj= subdir(1:is-1);
    filename= strcat(subdir, '/', file, '*');
    fprintf('* processing %s (%02d-%02d)\n', filename, li, lj);
    try,
      hdr= eegfile_readBVheader([filename(1:end-1) sbj]);
    catch,
      fprintf('not existing!\n');
      continue;
    end
    k= k+1;
    [cnt,mrk_orig]= eegfile_loadBV(filename, 'fs',100, 'clab',clab, ...
                                   'subsample_fcn', 'subsampleByLag');
    if length(chanind(cnt, 'PO3','PO4'))~=2,
      cidx= chanind(cnt, 'PO5','PO6');
      if isempty(cidx),
        cidx= chanind(cnt, 'PO1','PO2');
      end
      cnt.clab(cidx)= {'PO3','PO4'};
    end
    cnt= proc_selectChannels(cnt, 'not', 'PO1','PO2','PO5','PO6');
    if length(cnt.clab)~=45,
      fprintf('not all channels found\n');
      cnt.clab
      continue;
    end
    mrk= mrk_defineClasses(mrk_orig, {1, 2; 'left','right'});
    [filt_b,filt_a]= butter(5, [10 15]/cnt.fs*2);
    cnt= proc_filt(cnt, filt_b, filt_a);
    fv= cntToEpo(cnt, mrk, [750 3500]);

%% Validation with CSP inside a chron-fold crossvalidation
    proc= struct('memo', 'csp_w');
%    proc.train= ['[fv,csp_w]= proc_csp_auto(fv, 3); ' ...
    proc.train= ['[fv,csp_w]= proc_csp3(fv, 3); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv);'];
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv);'];
    [loss(k),loss_std(k)] = xvalidation(fv, 'LSR', opt_xv, 'proc',proc);
    hit(k)= 100-100*loss(k);
    save(sprintf('%s/baseline3_loss',export_dir), 'loss', 'loss_std');
  end
end
