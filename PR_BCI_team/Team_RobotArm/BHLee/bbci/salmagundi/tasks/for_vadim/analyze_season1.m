date_list= {'04_03_24', ...
            '04_03_25', ...
            '04_03_29', ...
            '04_03_30', ...
            '04_03_31', ...
            '04_04_08'};

feedback_files = {'imag_lett1Matthias_udp_1d_szenario_02',...
                  '',...
                  'imag_lettGuido_udp_1d_szenario_01',...
                  'imag_moveGabriel_udp_1d_szenario_02',...
                  'imag_lettFalk_udp_1d_szenario_01',...
                  'imag_lettKlaus_udp_1d_szenario_01',...
                  };

expbase= expbase_read;
imagtrain= expbase_select(expbase, 'paradigm',{'imag_lett', 'imag_move'});
imagtrain= expbase_joinParadigms(imagtrain, 'imag_lett', 'imag_move');

for ff= 1:length(date_list),
  %% load EEG data
  train_file= expbase_select(imagtrain, 'date',date_list{ff});
  file_name= expbase_filename(train_file);
  [cnt, mrk, mnt]= eegfile_loadMatlab(file_name);
  
  %% load set proprocessing settings
  szenario= feedback_files{ff};
  if isempty(szenario),
    %% no feedback was recorded for subject <ff>
    csp.band= [7 30];
    csp.ival= [750 4000];
    csp.clab= {'F7-8','FFC#','FC#','FT7,8','CFC#','C#','T7,8','CCP#', ...
               'CP#','TP7,8','PCP#','P7-8'};
    csp.nPat= 3;
    csp.usedPat= 4:6;
    csp.filtOrder= 5;
    [csp.filt_b, csp.filt_a]= butter(csp.filtOrder, csp.band/100*2);
    classes= {'left','right'};
  else
    %% load setting from actual feedback experiment
    S= load([EEG_RAW_DIR train_file.subject '_' train_file.date '/' szenario]);
    classes= S.classes;
    csp= S.csp;
    csp.filt_b= S.dscr.proc_cnt_apply.param{1};
    csp.filt_a= S.dscr.proc_cnt_apply.param{2};
  end
  
  cnt= proc_selectChannels(cnt, csp.clab);
    
    setup_plot_opts;

    %% plot spectra of imagery classes
    cnt_lap= proc_laplace(cnt, 'small', ' lap', 'E*');
    spec_lap= makeEpochs(cnt_lap, mrk, csp.ival);
    spec_lap= proc_spectrum(spec_lap, [5 35], 'db_scaled',1);
    grid_plot(spec_lap, mnt_spec, spec_opt);

    %% plot r^2 values of spectral differences between imagery classes
    rsq= proc_r_square(spec_lap);
    grid_plot(rsq, mnt_spec, spec_opt, rsqu_opt{:});
    grid_markIval(csp.band);
    clear spec_lap rsq
    
    %% calculate ERD curves of laplace channels
    cnt_flt= proc_filt(cnt_lap, csp.filt_b, csp.filt_a);
    cnt_flt.title= sprintf('%s  [%d %d] Hz', cnt.title, csp.band);
    erd= makeEpochs(cnt_flt, mrk, [-500 4500]);
    erd= proc_rectifyChannels(erd);
    erd= proc_movingAverage(erd, 150);
    %erd= proc_baseline(erd, [-250 250]);
    grid_plot(erd, mnt_spec, grid_opt, ...
              'xTick', 0:2000:4000, 'xTickLabelMode','auto');
    grid_markIval(csp.ival);
    
    %% calculate baseline spectrum
    %% (baseline = whole experiment without breaks for relaxation)
    if iscell(file_name),
      blk_ival= getActivationAreas(file_name{1});
      for qq= 2:length(file_name),
        blk_ival2= getActivationAreas(file_name{qq});
        blk_ival= cat(1, blk_ival, blk_ival2 + cnt.T(qq-1));
      end
    else
      blk_ival= getActivationAreas(cnt.title);
    end
    blk= struct('fs',cnt.fs, 'ival',blk_ival');
    [spec_base, blk]= proc_concatBlocks(cnt, blk);
    mk= mrk_evenlyInBlocks(blk, 1000);
    spec_base= makeEpochs(spec_base, mk, [0 990]);
    spec_base= proc_spectrum(spec_base, csp.band, ...
                             'win',hamming(cnt.fs), 'db_scaled',0);
    spec_base= proc_average(spec_base);
    spec_base.className= {'ref'};
    
    %% calculate spectra of each imagery class
    epo= makeEpochs(cnt, mrk, [-500 4000]);
    spec= proc_selectIval(epo, csp.ival);
    spec= proc_spectrum(spec, csp.band, ...
                        'win',hamming(cnt.fs), 'db_scaled',0);
%    plotClassTopographies(spec, mnt, csp.band, scalp_opt);

    %% plot topographies of spectral differences (imagery - baseline)
    dspec= proc_subtractReferenceClass(spec, spec_base);
    plotClassTopographies(dspec, mnt, csp.band, scalp_opt);
    clear spec
    
    %% plot r^2 topographies of spectral differences
    rsq= proc_r_square(dspec);
    hp= plotClassTopographies(rsq, mnt, csp.band, scalp_opt);
    pos= get(hp(2), 'position');
    pos(2)= pos(2) - 0.1;
    set(hp(2), 'position',pos);
    clear dspec rsq
    
    if isempty(feedback_files{ff}),
      continue;  %% no feedback was perform for subject <ff>
    end

    %% plot CSP scalp maps
    csp_w= S.dscr.proc_cnt_apply.param{3};
    nPat= size(csp_w,2);
    if nPat==6, nPat=nPat/2; end
    head= restrictDisplayChannels(mnt, csp.clab);
    plotCSPatterns(nPat, head, csp_w, [], scalp_opt);
    addTitle(untex(cnt.title), 1, 0);
 
    
    %% plot ERD curves of csp channels
    cnt_flt= proc_selectChannels(cnt, S.dscr.clab);
    cnt_flt= feval(S.dscr.proc_cnt_apply.fcn, cnt_flt, [], ...
                   S.dscr.proc_cnt_apply.param{:});
    cnt_flt.title= sprintf('%s  [%d %d] Hz', cnt.title, csp.band);
    erd= makeEpochs(cnt_flt, mrk, [-500 4500]);
    erd= proc_rectifyChannels(erd);
    erd= proc_movingAverage(erd, 150);
    %erd= proc_baseline(erd, [-250 250]);
    if length(erd.clab)==6,
      grd= sprintf('%s,_\n%s,legend', vec2str(erd.clab(1:3), '%s', ','), ...
                   vec2str(erd.clab(4:6), '%s', ','));
    else
      grd= [sprintf('%s,',erd.clab{:}) 'legend'];
    end
    mnt_csp= setDisplayMontage(erd.clab, grd);
    grid_plot(erd, mnt_csp, 'colorOrder',grid_opt.colorOrder, ...
                            'axisTitleFontWeight', 'bold');
    grid_markIval(csp.ival);
  
end
