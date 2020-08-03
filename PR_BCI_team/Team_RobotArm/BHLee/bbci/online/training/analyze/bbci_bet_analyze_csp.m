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
% Guido Dornhege, 07/12/2004
% $Id: bbci_bet_analyze_csp.m,v 1.10 2007/06/27 16:23:11 neuro_cvs Exp $


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
%analyze = struct;
analyze = [];

%TODO: DOCUMENTATION OF THIS SCRIPT

grd= sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
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

if ~isfield(opt, 'usedPat')
  opt.usedPat = 1:2*opt.nPat;
end

%% when nPat was changed, but usedPat was not, define usedPat
if isfield(bbci_bet_memo_opt, 'nPat') & bbci_bet_memo_opt.nPat~=opt.nPat ...
      & isequal(bbci_bet_memo_opt.usedPat, opt.usedPat),
  opt.usedPat = 1:2*opt.nPat;
end

if ~isfield(opt,'band') | isempty(opt.band);
  bbci_bet_message('No band specified, select an optimal one: ');
  if isfield(opt,'ival') & ~isempty(opt.ival)
    opt.band = select_band(Cnt,mrk_selectClasses(mrk,bbci.classes),opt.ival);
  elseif isfield(opt,'default_ival')
    opt.band = select_band(Cnt,mrk_selectClasses(mrk,bbci.classes), ...
			   opt.default_ival);
  else
    message_box('You should define opt.default_ival or opt.ival',1);
    return;
  end
  bbci_bet_message('[%i %i]\n',opt.band);
end

disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(Cnt, disp_clab);

[csp_b, csp_a]= butter(opt.filtOrder, opt.band/Cnt.fs*2);
cnt_flt= proc_filt(Cnt, csp_b, csp_a);
cnt_flt.title= sprintf('%s  [%d %d] Hz', Cnt.short_title, opt.band);
bbci_bet_message('Data filtered\n');  

if ~isfield(opt,'ival') | isempty(opt.ival)
  bbci_bet_message('No ival specified, selecting an optimal one: ');
  opt.ival = select_ival(cnt_flt,mrk_selectClasses(mrk,bbci.classes));
  bbci_bet_message('[%i %i]\n',opt.ival);
end


if bbci.withgraphics 
  epo= proc_selectChannels(Cnt, requ_clab);
  epo= makeEpochs(epo, mrk, opt.dar_ival);
  epo= proc_baseline(epo, [-500 0]);

  bbci_bet_message('Creating figure 1\n');
  handlefigures('use','Spectra');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: spectra in [%d %d] ms', ...
				      Cnt.short_title, opt.ival));
  spec= proc_selectIval(epo, opt.ival);
  spec= proc_laplace(spec);
  spec= proc_spectrum(spec, [5 35], epo.fs);
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
  erd= proc_laplace(erd);
  erd= proc_rectifyChannels(erd);
  erd= proc_movingAverage(erd, 200);
  erd= proc_baseline(erd, [-500 0]);
  epo_rsq= proc_r_square(proc_selectClasses(erd,bbci.classes));

  
  h = grid_plot(erd, mnt, grid_opt);
  grid_markIval(opt.ival);
  hh = hot(64); hh = hh(end:-1:1,:);
  hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_addBars(epo_rsq,'colormap',hh,'height',0.06,'h_scale',h.scale); 

  clear erd epo_rsq;
end

bbci_bet_message('Outlierness\n');
fig1 = handlefigures('use','trial-outlierness');
fig2 = handlefigures('use','channel-outlierness');

mrk_cl= mrk_selectClasses(mrk, bbci.classes);
cnt_flt= proc_selectChannels(cnt_flt, opt.clab{:});
fv= makeEpochs(cnt_flt, mrk_cl, opt.ival);

if isfield(opt, 'threshold') & opt.threshold<inf
  fv = proc_outl_var(fv, 'trialthresh',opt.threshold,...
		     'display',bbci.withclassification,...
		     'handles',[fig1,fig2]);
else
  proc_outl_var(fv, 'display',bbci.withclassification,...
		'handles',[fig1,fig2]);
end

bbci_bet_message('CSP\n');
%% Vorsicht, Hacker unterwegs
global hlp_w
[fv2, hlp_w, la, A]= proc_csp3(fv, 'patterns',opt.nPat, 'scaling','maxto1');


clear cnt_flt

if bbci.withgraphics | bbci.withclassification,
  bbci_bet_message('Creating Figure CSP\n');
  handlefigures('use','CSP');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: CSP <%s> vs <%s>', Cnt.short_title, ...
			  bbci.classes{:}));
  plotCSPanalysis(fv, mnt, hlp_w, A, la, 'mark_patterns', opt.usedPat);
end

bbci_bet_message('Calculate features\n');
features= proc_variance(fv2);
features= proc_logarithm(features);
features.x = features.x(:,opt.usedPat,:);
hlp_w = hlp_w(:,opt.usedPat);

if bbci.withclassification,
  opt_xv= strukt('sample_fcn',{'chronKfold',8}, 'std_of_means',0);
  [loss,loss_std] = xvalidation(features, opt.model, opt_xv);
  bbci_bet_message('CSP global: %2.1f +/- %1.1f\n',100*loss,100*loss_std);
  remainmessage = sprintf('CSP global: %3.1f +/- %3.1f', 100*loss,100*loss_std);
  proc= struct('memo', 'csp_w');
  proc.train= ['[fv,csp_w]= proc_csp3(fv, ' int2str(opt.nPat) '); ' ...
	       'fv= proc_variance(fv); ' ...
	       'fv= proc_logarithm(fv);'];
  proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
	       'fv= proc_variance(fv); ' ...
	       'fv= proc_logarithm(fv);'];
  [loss,loss_std] = xvalidation(fv, opt.model, opt_xv, 'proc',proc);
  bbci_bet_message('CSP inside: %2.1f +/- %1.1f\n',100*loss,100*loss_std);
  remainmessage = sprintf('%s\nCSP inside: %3.1f +/- %3.1f', ...
			  remainmessage, 100*loss,100*loss_std);
  if ~isequal(opt.usedPat, 1:2*opt.nPat),
    proc.train= ['global hlp_w; ' ...
		 '[fv,csp_w]= proc_csp3(fv, ''patterns'',hlp_w, ' ...
		 '''selectPolicy'',''matchfilters''); ' ...
		 'fv= proc_variance(fv); ' ...
		 'fv= proc_logarithm(fv);'];
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
		 'fv= proc_variance(fv); ' ...
		 'fv= proc_logarithm(fv);'];
    [loss,loss_std] = xvalidation(fv, opt.model, opt_xv, 'proc',proc);
    bbci_bet_message('CSP selPat: %2.1f +/- %1.1f\n',100*loss,100*loss_std);
    remainmessage = sprintf('%s\nCSP setPat: %3.1f +/- %3.1f', ...
			    remainmessage, 100*loss,100*loss_std);
  end
else
  remainmessage = 'No classification was performed.';
end


% What do we need later?
analyze = struct('csp_a', csp_a, 'csp_b', csp_b, 'csp_w', hlp_w, ...
		 'features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
