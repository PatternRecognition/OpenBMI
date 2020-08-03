%BBCI_BET_ANALYZE_CSPAUTO - Analysis of SMR Modulations for CSP-based BBCI
%
%Description.
% Analyzes data provided by bbci_bet_prepare according to the
% parameters specified in the struct 'bbci'.
% It provides features for bbci_bet_finish_cspauto
%
%Input (variables defined before calling this function):
% Cnt, mrk, mnt:  (loaded by bbci_bet_prepare)
% bbci:   struct, the general setup variable from bbci_bet_prepare
% bbci_memo: internal use
% opt    (copied by bbci_bet_analyze from bbci.setup_opts) a struct with fields
%  reject_artifacts:
%  reject_channels:
%  reject_opts: cell array of options which are passed to
%     reject_varEventsAndChannels.
%  reject_outliers:
%  check_ival: interval which is checked for artifacts/outliers
%  ival: interval on which CSP is performed, 'auto' means automatic selection.
%  band: frequency band on which CSP is performed, 'auto' means
%     automatic selection.
%  nPat: number of CSP patterns which are considered from each side of the
%     eigenvalue spectrum. Note that not neccessarily all of these are
%     used for classification, see opt.usedPat.
%  usedPat: vector specifying the indices of the CSP filters that should
%     be used for classification, 'auto' means automatic selection.
%  do_laplace: do Laplace spatial filtering for automatic selection
%     of ival/band. If opt.do_laplace is set to 0, the default value of
%     will also be set to opt.visu_laplace 0 (but can be overwritten).
%  visu_laplace: do Laplace filtering for grid plots of Spectra/ERDs.
%     If visu_laplace is a string, it is passed to proc_laplace. This
%     can be used to use alternative geometries, like 'vertical'.
%  visu_band: Frequency range for which spectrum is shown.
%  visu_ival: Time interval for which ERD/ERS curves are shown.
%  visu_classes: Classes for which Spectra and ERD curves are drawn,
%     default '*'.
%  grd: grid to be used in the grid plots of spectra and ERDs.
%
%Output:
%   analyze  struct, will be passed on to bbci_bet_finish_csp
%   bbci : updated
%   bbci_memo : updated

% blanker@cs.tu-berlin.de, Aug-2007
% Guido Dornhege, 07/12/2004


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
%analyze = struct;
analyze = [];








%TODO: DOCUMENTATION OF THIS SCRIPT

%default_grd= ...
%    sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');


grd_std= sprintf(['scale,Fp1,AF3,AF4, Fp2,legend\n' ...
    'F5,F3,F1,Fz,F2,F4,F6\n'...
    'FC5,FC3,FC1,FCz,FC2,FC4,FC6\n' ...
    'C5,C3,C1,Cz,C2,C4,C6\n' ...
    'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
    'P5,P3,P1,Pz,P2,P4,P6\n']);


grd_mcc= sprintf(['<,scale,Fp1,AF3,AF4,Fp2,legend\n' ...
    'F5,F3,F1,Fz,F2,F4,F6\n'...
    'FC5,FC3,FC1,FCz,FC2,FC4,FC6\n' ...
    '<,CFC5,CFC3,CFC1,CFC2,CFC4,CFC6\n' ...
    'C5,C3,C1,Cz,C2,C4,C6\n' ...
    '<,CCP5,CCP3,CCP1,CCP2,CCP4,CCP6\n' ...
    'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
    'P5,P3,P1,Pz,P2,P4,P6\n']);

default_grd = grd_mcc;
default_colDef= {'left', 'right',   'foot',  'rest'; ...
    [0.8 0 0.8], [0 0.7 0], [0 0 1], [0 0 0]};

[opt, isdefault]= ...
    set_defaults(opt, ...
    'reject_artifacts', 1, ...
    'reject_channels', 1, ...
    'reject_opts', {}, ...
    'reject_outliers', 0, ...
    'check_ival', [500 4500], ...
    'ival', 'auto', ...
    'default_ival', [1000 3500], ...
    'min_ival_length', 300, ...       % changed!
    'enlarge_ival_append', 'end', ...  % changed!
    'repeat_bandselection', 1, ...
    'selband_opt', [], ...
    'selival_opt', [], ...
    'usedPat', 'auto', ...
    'do_laplace', 1, ...
    'visu_laplace', 1, ...
    'visu_band', [5 35], ...
    'visu_ival', [-500 10000], ...
    'visu_classes', '*', ...
    'grd', default_grd, ...
    'colDef', default_colDef, ...
    'verbose', 1);
%% TODO: optional visu_specmaps, visu_erdmaps

bbci_bet_memo_opt= ...
    set_defaults(bbci_bet_memo_opt, ...
    'nPat', NaN, ...
    'usedPat', NaN, ...
    'band', NaN);

if isdefault.default_ival & opt.verbose,
    msg= sprintf('default ival not defined in bbci.setup_opts, using [%d %d]', ...
        opt.default_ival);
    warning(msg);
end
if isdefault.visu_laplace & ~opt.do_laplace,
    opt.visu_laplace= 0;
end



%% Prepare visualization

mnt= mnt_setGrid(mnt, opt.grd);
opt_grid= defopt_erps('scale_leftshift',0.075);
%% TODO: extract good channel (like 'CPz' here) from grid
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
    'xTickAxes','CPz');
clab_gr1= intersect(scalpChannels, getClabOfGrid(mnt));
if isempty(clab_gr1),
    clab_gr1= getClabOfGrid(mnt);
end
opt_grid.scaleGroup= {clab_gr1, {'EMG*'}, {'EOG*'}};
fig_opt= {'numberTitle','off', 'menuBar','none'};
if length(strpatternmatch(mrk.className, opt.colDef(1,:))) < ...
        length(mrk.className),
    if ~isdefault.colDef,
        warning('opt.colDef does not match with mrk.className');
    end
    nClasses= length(mrk.className);
    cols= mat2cell(cmap_rainbow(nClasses), ones(1,nClasses), 3)';
    opt.colDef= {mrk.className{:}; cols{:}};
end

%% when nPat was changed, but usedPat was not, define usedPat
if bbci_bet_memo_opt.nPat~=opt.nPat ...
        & ~strcmpi(opt.usedPat, 'auto') ...
        & isequal(bbci_bet_memo_opt.usedPat, opt.usedPat),
    opt.usedPat= 1:2*opt.nPat;
end


%% Analysis starts here
clear fv*
clab= Cnt.clab(chanind(Cnt, opt.clab));

%% artifact rejection (trials and/or channels)
flds= {'reject_artifacts', 'reject_channels', ...
    'reject_opts', 'check_ival', 'clab'};
if bbci_memo.data_reloaded | ...
        ~fieldsareequal(bbci_bet_memo_opt, opt, flds),
    clear anal
    anal.rej_trials= NaN;
    anal.rej_clab= NaN;
    if opt.reject_artifacts | opt.reject_channels,
        if opt.verbose,
            fprintf('checking for artifacts and bad channels\n');
        end
        if bbci.withgraphics,
            handlefigures('use', 'Artifact rejection', 1);
            set(gcf, fig_opt{:},  ...
                'name',sprintf('%s: Artifact rejection', Cnt.short_title));
        end
        [mk_clean , rClab, rTrials]= ...
            reject_varEventsAndChannels(Cnt, mrk, opt.check_ival, ...
            'do_multipass', 1, ...
            opt.reject_opts{:}, ...
            'visualize', bbci.withgraphics, ...
            'clab',clab); % Achtung nur Ausgewï¿½hlte Channels
        if bbci.withgraphics,
            handlefigures('next_fig','Artifact rejection');
            drawnow;
        end
        if opt.reject_artifacts,
            if length(rTrials)>0 | opt.verbose,
                %% TODO: make output class-wise
                fprintf('rejected: %d trial(s).\n', length(rTrials));
            end
            anal.rej_trials= rTrials;
        end
        if opt.reject_channels,
            if length(rClab)>0 | opt.verbose,
                fprintf('rejected channels: <%s>.\n', vec2str(rClab));
            end
            anal.rej_clab= rClab;
        end
    end
end
if iscell(anal.rej_clab),   %% that means anal.rej_clab is not NaN
    clab(strpatternmatch(anal.rej_clab, clab))= [];
end

if opt.reject_outliers,
    %% TODO: execute only if neccessary
    if opt.verbose,
        bbci_bet_message('checking for outliers\n');
    end
    fig1 = handlefigures('use', 'trial-outlierness', 1);
    fig2 = handlefigures('use', 'channel-outlierness', 1);
    %% TODO: reject_outliers only on artifact free trials?
    %%  clarify relation of reject_articfacts and reject_outliers
    fv= cntToEpo(Cnt, mk, opt.check_ival, 'clab',clab);
    [fv, anal.outl_trials]=  ...
        proc_outl_var(fv, ...
        'display', bbci.withclassification,...
        'handles', [fig1,fig2], ...
        'trialthresh',bbci.setup_opts.threshold);
    %% TODO: output number of outlier trials (class-wise)
    clear fv
    handlefigures('next_fig', 'trial-outlierness');
    handlefigures('next_fig', 'channel-outlierness');
else
    anal.outl_trials= NaN;
end

kickout_trials= union(anal.rej_trials, anal.outl_trials);
kickout_trials(find(isnan(kickout_trials)))= [];
this_mrk= mrk_chooseEvents(mrk, setdiff(1:length(mrk.pos), kickout_trials));


if ~isequal(bbci.classes, 'auto'),
    class_combination= strpatternmatch(bbci.classes, this_mrk.className);
    if length(class_combination) < length(bbci.classes),
        error('not all specified classes found');
    end
    if opt.verbose,
        fprintf('using classes <%s> as specified\n', vec2str(bbci.classes));
    end
else
    class_combination= nchoosek(1:size(this_mrk.y,1), 2);
end


%% Specific investigation of binary class combination(s) start
memo_opt.band= opt.band;
memo_opt.ival= opt.ival;
clear analyze mean_loss std_loss
for ci= 1:size(class_combination,1),

    classes= this_mrk.className(class_combination(ci,:));
    if strcmp(classes{1},'right') | strcmp(classes{2},'left'),
        class_combination(ci,:)= fliplr(class_combination(ci,:));
        classes= this_mrk.className(class_combination(ci,:));
    end
    if size(class_combination,1)>1,
        fprintf('\ninvestigating class combination <%s> vs <%s>\n', classes{:});
    end
    mrk2= mrk_selectClasses(this_mrk, classes);
    opt_grid.colorOrder= choose_colors(this_mrk.className, opt.colDef);
    opt_grid.lineStyleOrder= {'--','--','--'};
    clidx= strpatternmatch(classes, this_mrk.className);
    opt_grid.lineStyleOrder(clidx)= {'-'};
    opt_grid_spec.lineStyleOrder= opt_grid.lineStyleOrder;
    opt_grid_spec.colorOrder= opt_grid.colorOrder;

    bbci_bet_message('LRP Filter');

    Cnt =  proc_commonAverageReference(Cnt,scalpChannels(Cnt));

    %opt.band = [0.1,6];
    %Wps= [4 6]/Cnt.fs*2;
    %db_attenuation=60;
    %[n, Wn]= cheb2ord(Wps(1), Wps(2), 3, db_attenuation);

    %[filt_b,filt_a]= butter(opt.filtOrder, opt.band/Cnt.fs*2);
    %cnt_flt= proc_filt(cnt_flt, filt_b, filt_a);

    % [filt_b, filt_a]= cheby2(n, db_attenuation, Wn);
    % cnt_flt= proc_filt(Cnt, filt_b, filt_a);

    % Hier müsste man opt.band und filtOrd verwenden statt diesen fix
    % codierten Werten
    Wps= [6 7]/Cnt.fs*2;
    db_attenuation=30;
    [n, Wn]= cheb2ord(Wps(1), Wps(2), 3, db_attenuation);
    [filt_b, filt_a]= cheby2(n, db_attenuation, Wn);
    Cnt= proc_filt(Cnt, filt_b, filt_a);
    % Die filt_b und filt_a müssten noch an bbci.filt.b und bbci.filt.a
    % übergeben werden.

    epo= cntToEpo(Cnt, mrk2, opt.ival, 'clab',clab);
    %epo_r= proc_r_square_signed(epo);
    clear cnt_flt

    %% Visualization of LRP

    if bbci.withgraphics
        disp_clab= getClabOfGrid(mnt);
        requ_clab= disp_clab;

        if opt.verbose>=2,
            bbci_bet_message('Creating figures.... \n');
        end

        %{
  handlefigures('use','Spectra',1);
  set(gcf, fig_opt{:}, 'name',...
           sprintf('%s: spectra in [%d %d] ms', Cnt.short_title, opt.ival));

    Win = Cnt.fs;

  spec= proc_spectrum(epo, opt.visu_band, kaiser(Win,2));
  spec_rsq= proc_r_square_signed(proc_selectClasses(spec,classes));

  h= grid_plot(spec, mnt, opt_grid_spec);
  grid_markIval(opt.band);
  grid_addBars(spec_rsq, ...
               'h_scale', h.scale, ...
               'box', 'on', ...
               'colormap', cmap_posneg(31), ...
               'cLim', 'sym');
  drawnow;

  clear spec_rsq spec
  handlefigures('next_fig','Spectra');
  drawnow;
        %}

        if opt.verbose>=2,
            bbci_bet_message('Creating figure(s) for LRP\n');
        end
        epo = proc_baseline(epo, opt.baseline);
        
        epo_r= proc_r_square_signed(proc_selectClasses(epo, classes));
        handlefigures('use','INTERVAL');

        % Heuristic: Find good time intervals
        opt.selectival = select_time_intervals(proc_selectIval(epo_r, [300 opt.ival(2)-1000]), 'nIvals',1, 'visualize', 1, 'visu_scalps', 0);

        % Settings for VPfbc
        opt.selectival = [600 2500]
        %opt.selectival = [0:500:4000]
        
        handlefigures('use','LRP',size(opt.band,1));
        set(gcf, fig_opt{:},  ...
            'name',sprintf('%s: ERD for [%g %g] Hz', ...
            Cnt.short_title, opt.band));

        h = grid_plot(epo, mnt, opt_grid);
        grid_markIval(opt.selectival);
        grid_addBars(epo_r, ...
            'h_scale',h.scale, ...
            'box', 'on', ...
            'colormap', cmap_posneg(31), ...
            'cLim', 'sym');
        drawnow;
        %set(h,'MenuBar','figure');
        handlefigures('next_fig','LRP');

        myH=handlefigures('use','ScalpPicture')
        scalpEvolutionPlusChannel(epo, mnt, 'C3' , opt.selectival, 'legend_pos',2 , 'extrapolate', 1,'shading','interp','extrapolateToMean',1, 'globalCLim', 1, 'renderer', 'contourf');
        set(myH,'MenuBar','figure');

        myH=handlefigures('use','ScalpPictureR');
        scalpEvolutionPlusChannel(epo_r, mnt, 'C3' , opt.selectival, 'legend_pos',2 , 'extrapolate', 1,'shading','interp','extrapolateToMean',1, 'globalCLim', 1, 'renderer', 'contourf');
        set(myH,'MenuBar','figure');
        %{
  Fig1 = figure;
  set(Fig1,'Visible', 'off');
  plotChannel(lrp,'C4');
  grid_markIval(ival_list);
  Fig2 = figure;
  set(Fig2,'Visible', 'off');
  scalpEvolution(lrp, mnt, ival_list, 'legend_pos',2 , 'extrapolate', 1,'shading','interp','extrapolateToMean',1, 'globalCLim', 1, 'renderer', 'contourf');
  Fig3 = figure;
  set(Fig3,'Visible', 'off');
  scalpEvolution(lrp_rsq, mnt, ival_list, 'legend_pos',2 , 'extrapolate', 1,'shading','interp','extrapolateToMean',1, 'globalCLim', 1, 'renderer', 'contourf');

  handlefigures('use','ScalpPicture');
  H = fig2subplot([Fig1,Fig2,Fig3],'rowscols',[3 1],'deleteFigs',1);
        %}

        %clear lrp lrp_rsq;
    end


    %% ToDO: Check with DAVID: Ist es korrekt, hier wieder mit epo
    %% rumzumachen statt mit dem lrp????
    if opt.verbose>=2,
        bbci_bet_message('calculating LRP\n');
    end

    %opt.selectival = select_time_intervals(proc_selectIval(epo_r, [500 opt.ival(2)-1000]), 'nIvals',1, 'visualize', 0, 'sort', 1);
    %epo= proc_baseline(epo, opt.baseline);
    
    fv= proc_jumpingMeans(epo, opt.selectival); % chance to mean

    % Noch noetig?? Vermutlich nicht fuer LRP...
    %[fv, opt.meanOpt] = proc_subtractMean(proc_flaten(fv));
    %[fv, opt.normOpt] = proc_normalize(fv);



    %% BB: I propose a different validation. For test samples always take a
    %%  FIXED interval, e.g. opt.default_ival. Practically this can be done
    %%  with bidx, train_jits, test_jits.
    remainmessage = '';
    if bbci.withclassification,
        
        % opt_xv= strukt('xTrials', [5 5]);
        
        % skalierung der Ausgabe auf 1 und store_means ist bereits im run
        % script run_Fissler aktiviert worden, daher dies unnütz:
        % classy={opt.model,  struct('scaling',1)}
        opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
            'std_of_means',0, ...
            'verbosity',0, ...
            'progress_bar',0);
        [loss,loss_std] = xvalidation(fv, opt.model, opt_xv);
        bbci_bet_message('LRP Common Average: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
%         remainmessage= sprintf('CSP global: %4.1f +/- %3.1f',100*loss,100*loss_std);
% 
%         opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
%             'std_of_means',0, ...
%             'verbosity',0, ...
%             'progress_bar',0);
% 
% 
%         [loss,loss_std] = xvalidation(fv,'FDshrink', opt_xv);
%         bbci_bet_message('LRP Common Average: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
%         remainmessage= sprintf('CSP global: %4.1f +/- %3.1f', ...
%             100*loss,100*loss_std);
        %{
class.classModel = 'FDshrink';
class.proc = struct('memo', 'passParams');

class.proc.train= ['passParams=select_time_intervals(proc_r_square_signed(fv), ''nIvals'',2, ''ival_pick_peak'', [500 5000]);fv=proc_jumpingMeans(fv,passParams);'];
class.proc.apply= ['fv= proc_jumpingMeans(fv, passParams); '];
class.sample_fcn = {'chronKfold', 10};
class.ms_sample_fcn = {'chronKfold', 10};
class.outer_ms= 1;
class.save_proc_params = {'passParams'};
class.save_classifier = 1;
           class.loss = {'classwiseNormalized', sum(epo.y,2)};
   [loss,loss_std] = xvalidation(epo, class.classModel, class);
  bbci_bet_message('LRP Common Average: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
  remainmessage= sprintf('CSP global: %4.1f +/- %3.1f', 100*loss,100*loss_std);
            %}

    end
    mean_loss(ci)= loss;
    std_loss(ci)= loss_std;
end
% Gather all information that should be saved in the classifier file
analyze(ci)= merge_structs(anal, strukt('clab', clab, ...
    'csp_a', filt_a, ...
    'csp_b', filt_b, ...
    'features', fv, ...
    'ival', opt.ival, ...
    'band', opt.band, ...
    'baseline', opt.baseline, ...
    'message', remainmessage));



if isequal(bbci.classes, 'auto'),
    [dmy, bi]= min(mean_loss + 0.1*std_loss);
    bbci.classes= this_mrk.className(class_combination(bi,:));
    bbci.class_selection_loss= [mean_loss; std_loss];
    analyze= analyze(bi);
    bbci_bet_message(sprintf('\nCombination <%s> vs <%s> chosen.\n', ...
        bbci.classes{:}));
    if bi<size(class_combination,1),
        opt.ival= analyze.ival;    %% restore selection of best class combination
        opt.band= analyze.band;
        bbci_bet_message('Rerun bbci_bet_analyze again to see corresponding plots\n');
    end
end
bbci_memo.data_reloaded= 0;

if opt.verbose>=2,
    bbci_bet_message('Finished analysis\n');
end