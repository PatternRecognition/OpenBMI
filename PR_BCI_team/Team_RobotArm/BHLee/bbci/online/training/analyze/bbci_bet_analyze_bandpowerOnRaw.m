% !! this is only a quick hack so far !!
%
%BBCI_BET_ANALYZE_* ANALYZES THE DATA PROVIDED BY
%BBCI_BET_PREPARE AND THE SPECIFIC SUBJECT FILE BY MEANS OF PLOTS
%AND CLASSIFICATION PERFORMANCES. FURTHERMORE IT PROVIDES THE
%FEATURES FOR BBCI_BET_FINISH_*.
% 
% input:
%   bbci   struct, the general setup variable from bbci_bet_prepare
%   opt    a struct with fields
%       clab    the used channels
%       ival    the train interval
%       band    the used band
%       filtOrder   the filt Order
%       ilen_apply the apply length 
%       dar_ival    the ival the plots are visulized
%       model   the model
%   Cnt, mrk, mnt
%
% output:
%   analyze  struct, will be passed on to bbci_bet_finish_*
% 
% Guido Dornhege, 07/12/2004  -> Benjamin 10/2006


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
%analyze = struct;
analyze = [];

%TODO: DOCUMENTATION OF THIS SCRIPT

grd= sprintf('C3,Cz,C4,legend\nCP3,CPz,CP4,scale\nO1,Oz,O2,_');
mnt= mnt_setGrid(mnt, grd);
mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;

colOrder= [[1 0 0];[0 0.7 0];[0 0 1];[0 1 1];[1 0 1]; [1 1 0]];
grid_opt= struct('colorOrder', colOrder);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};

spec_opt= grid_opt;
spec_opt.yUnit= 'power';
spec_opt.xTickMode= 'auto';
spec_opt.xTick= 10:5:30;
spec_opt.xTickLabelMode= 'auto';
rsqu_opt= {'colorOrder','rainbow'};

fig_opt= {'numberTitle','off', 'menuBar','none'};


opt= set_defaults(opt, 'filt_b', [], 'filt_a',[], 'proc_fs',bbci.fs);
if ~isempty(opt.filt_b),
  Ct= proc_filt(Cnt, opt.filt_b, opt.filt_a);
else
  Ct= Cnt;
end
mk= mrk;
if opt.proc_fs~=bbci.fs,
  [Ct, mk]= proc_subsampleByLag(Ct, bbci.fs/opt.proc_fs, mrk);
end

if ~isfield(opt,'band') | isempty(opt.band);
  bbci_bet_message('No band specified, select an optimal one [NOLAP!]: ');
  if isfield(opt,'ival') & ~isempty(opt.ival),
    ival= opt.ival;
  elseif isfield(opt,'default_ival')
    ival= opt.default_ival;
  else
    message_box('You should define opt.default_ival or opt.ival',1);
    return;
  end
  opt.band = select_bandnarrow(Ct, ...
                               mrk_selectClasses(mk,bbci.classes), ...
                               ival, ...
                               'do_laplace',0);
  bbci_bet_message('[%g %g]\n',opt.band);
end

disp_clab= getClabOfGrid(mnt);
requ_clab= disp_clab;

[csp_b, csp_a]= butter(opt.filtOrder, opt.band/Ct.fs*2);
cnt_flt= proc_filt(Ct, csp_b, csp_a);
cnt_flt.title= sprintf('%s  [%d %d] Hz', Ct.short_title, opt.band);
bbci_bet_message('Data filtered\n');  

if ~isfield(opt,'ival') | isempty(opt.ival)
  bbci_bet_message('No ival specified, selecting an optimal one [NOLAP!]: ');
  opt.ival = select_timeival(cnt_flt, ...
                             mrk_selectClasses(mk,bbci.classes), ...
                             'do_laplace',0);
  bbci_bet_message('[%i %i]\n',opt.ival);
end


if bbci.withgraphics 
  epo= proc_selectChannels(Ct, requ_clab);
  epo= makeEpochs(epo, mk, opt.dar_ival);
  epo= proc_baseline(epo, [-500 0]);

  bbci_bet_message('Creating figure 1\n');
  handlefigures('use','Spectra');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: spectra in [%d %d] ms', ...
				      Ct.short_title, opt.ival));
  spec= proc_selectIval(epo, opt.ival);
  spec= proc_spectrum(spec, [5 35], epo.fs);
  epo_rsq= proc_r_square(proc_selectClasses(spec,bbci.classes));

  h = grid_plot(spec, mnt_spec, spec_opt);
  grid_markIval(opt.band);
  hh= flipud(hot(64));
  colormap(hh);
  grid_addBars(epo_rsq,'colormap',hh,'height',0.12,'h_scale',h.scale);
  
  clear epo epo_rsq;  % spec
end
%clear Ct


if bbci.withgraphics
  bbci_bet_message('Creating figure 2\n');
  handlefigures('use','ERD');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: ERD in [%g %g] Hz', Cnt.short_title, opt.band));
  erd= makeEpochs(cnt_flt, mk, opt.dar_ival);
  erd= proc_rectifyChannels(erd);
  erd= proc_movingAverage(erd, 200);
  erd= proc_baseline(erd, [-500 0]);
  epo_rsq= proc_r_square(proc_selectClasses(erd,bbci.classes));
  h = grid_plot(erd, mnt, grid_opt);
  grid_markIval(opt.ival);
  hh= flipud(hot(64));
  colormap(hh);
  grid_addBars(epo_rsq,'colormap',hh,'height',0.06,'h_scale',h.scale); 

  clear erd epo_rsq;
end

bbci_bet_message('Outlierness\n');
fig1 = handlefigures('use','trial-outlierness');
fig2 = handlefigures('use','channel-outlierness');

mrk_cl= mrk_selectClasses(mk, bbci.classes);
cnt_flt= proc_selectChannels(cnt_flt, opt.clab{:});
fv= makeEpochs(cnt_flt, mrk_cl, opt.ival);
clear cnt_flt

if isfield(opt, 'threshold') & isequal(opt.threshold, 'auto'),
  fv= proc_outl_var(fv, ...
                    'percentiles', [20 80], ...
                    'log', 1, ...
                    'handles',[fig1,fig2]);
elseif isfield(opt, 'threshold') & opt.threshold<inf
  fv = proc_outl_var(fv, ...
                     'log', 1, ...
                     'trialthresh',opt.threshold,...
                     'handles',[fig1,fig2]);
else
  proc_outl_var(fv, ...
                'percentiles', [20 80], ...
                'log', 1, ...
                'handles',[fig1,fig2]);
end

bbci_bet_message('Calculate features\n');
features= proc_variance(fv);
clear fv
features= proc_logarithm(features);

if bbci.withgraphics,
  bbci_bet_message('Creating Figure BandPower Topography\n');
  handlefigures('use','BandPowerTopography');
  set(gcf, fig_opt{:},  ...
           'name',sprintf('%s: BandPower <%s> vs <%s>', Cnt.short_title, ...
                          bbci.classes{:}));
  scalpPatterns(features, mnt, [], 'colAx','range');
end

opt_xv= strukt('sample_fcn',{'chronKfold',8}, 'std_of_means',0);
[loss,loss_std] = xvalidation(features, opt.model, opt_xv);
bbci_bet_message('xval: %2.1f +/- %1.1f\n',100*loss,100*loss_std);
remainmessage = sprintf('xval: %3.1f +/- %3.1f', 100*loss,100*loss_std);


% What do we need later?
analyze = struct('csp_a', csp_a, 'csp_b', csp_b, ...
                 'filt_b',opt.filt_b, 'filt_a',opt.filt_a, ...
                 'fs',bbci.fs, 'proc_fs',opt.proc_fs, ...
                 'features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
