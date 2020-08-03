%BBCI_BET_ANALYZE_CSP ANALYZES THE DATA PROVIDED BY
%BBCI_BET_PREPARE AND THE SPECIFIC SUBJECT FILE BY MEANS OF PLOTS
%AND CLASSIFICATION PERFORMANCES. FURTHERMORE IT PROVIDES THE
%FEATURES FOR BBCI_BET_FINISH_CSP. THIS IS DONE FOR THE CSP FEATURE
% 
% input:
%   bbci   struct, the general setup variable from bbci_bet_prepare
%   opt    a struct with fields
%       nPat    the number of patterns to use
%       usedPat the used Pattern (only for choose of the features)
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
%   analyze  struct, will be passed on to bbci_bet_finish_csp
%

analyze = [];

%TODO: DOCUMENTATION OF THIS SCRIPT

grd= sprintf('scale,FCz_CFC1,FCz_Fz,FCz_CFC2,legend\nC1_CFFC3,C1_CFC1,_,C2_CFC2,C2_CFC4\nC3_CCP3,C1_CCP3,_,C2_CCP4,C4_CCP4');
mnt= mnt_setGrid(mnt, grd);
mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;

colOrder= [[1 0 0];[0 0.7 0];[0 0 1];[0 1 1];[1 0 1]; [1 1 0]];
grid_opt= struct('colorOrder', colOrder);
%grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};

spec_opt= grid_opt;
spec_opt.yUnit= 'power';
spec_opt.xTickMode= 'auto';
spec_opt.xTick= 10:5:30;
spec_opt.xTickLabelMode= 'auto';
spec_opt.scalePolicy='individual';
spec_opt.NIRS=1;
rsqu_opt= {'colorOrder','rainbow'};

fig_opt= {'numberTitle','off', 'menuBar','none'};

if ~isfield(opt, 'usedPat')
  opt.usedPat = 1:2*opt.nPat;
end


[b, a]= butter(opt.filtOrder, opt.band/Cnt.fs*2);
cnt_flt= proc_filt(Cnt, b, a);
cnt_flt.title= sprintf('%s  [%d %d] Hz', Cnt.short_title, opt.band);
bbci_bet_message('Data filtered\n');  


if bbci.withgraphics 
  epo= Cnt;
  epo= makeEpochs(epo, mrk, opt.dar_ival);
  epo= proc_baseline(epo, [-2000 0]);

  bbci_bet_message('Creating figure 1\n');
  handlefigures('use','Spectra');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: spectra in [%d %d] ms', ...
				      Cnt.short_title, opt.ival));
  spec= proc_selectIval(epo, opt.default_ival);
  spec= proc_spectrum(spec, [1 3], 4*epo.fs);
  epo_rsq= proc_r_square(proc_selectClasses(spec,bbci.classes));
  
  h = grid_plot(spec, mnt_spec, spec_opt);
  grid_markIval(opt.band);
  hh = hot(64); hh = hh(end:-1:1,:);
  hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_addBars(epo_rsq,'colormap',hh,'height',0.12,'h_scale',h.scale);
  clear epo epo_rsq;  % spec
end



if bbci.withgraphics
  bbci_bet_message('Creating figure 2\n');
  handlefigures('use','ERD');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: ERD in [%d %d] Hz', Cnt.short_title, opt.band));
  erd= makeEpochs(cnt_flt, mrk, opt.dar_ival);
  %erd= proc_laplace(erd);
  %erd= proc_rectifyChannels(erd);
  erd= proc_movingAverage(erd, 200);
  erd= proc_baseline(erd, [-2000 0]);
  epo_rsq= proc_r_square(proc_selectClasses(erd,bbci.classes));
  h = grid_plot(erd, mnt, grid_opt);
  %grid_markIval(opt.ival);
  hh = hot(64); hh = hh(end:-1:1,:);
  hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_addBars(epo_rsq,'colormap',hh,'height',0.06,'h_scale',h.scale); 

  clear erd epo_rsq;
end

bbci_bet_message('Outlierness\n');
%fig1 = handlefigures('use','trial-outlierness');
%fig2 = handlefigures('use','channel-outlierness');

mrk_cl=mrk;
cnt_flt= proc_selectChannels(cnt_flt, opt.clab{:});
fv= makeEpochs(cnt_flt, mrk_cl, opt.default_ival);
% 
% if isfield(opt, 'threshold') & opt.threshold<inf
%   fv = proc_outl_var(fv, 'trialthresh',opt.threshold,...
% 		     'display',bbci.withclassification,...
% 		     'handles',[fig1,fig2]);
% else
%   proc_outl_var(fv, 'display',bbci.withclassification,...
% 		'handles',[fig1,fig2]);
% end

bbci_bet_message('CSP\n');
%% Vorsicht, Hacker unterwegs

features= fv;
features.x=mean(fv.x,1);

if bbci.withclassification,
  opt_xv= strukt('sample_fcn',{'chronKfold',8}, 'std_of_means',0);
  [loss,loss_std] = xvalidation(features, opt.model, opt_xv);
  bbci_bet_message('NIRS global: %2.1f +/- %1.1f\n',100*loss,100*loss_std);
  remainmessage = sprintf('NIRS global: %3.1f +/- %3.1f', 100*loss,100*loss_std);
 else
  remainmessage = 'No classification was performed.';
end


% What do we need later?
analyze = struct('a', a, 'b', b, ...
		 'features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
