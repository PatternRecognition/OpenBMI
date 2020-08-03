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
%       multicspmethod
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
% $Id: bbci_bet_analyze_cspmulti.m,v 1.1 2006/04/27 14:24:38 neuro_cvs Exp $


% Everything that should be carried on to bbci_bet_finish_csp must go
% into a variable of this name:
analyze = struct;

%TODO: DOCUMENTATION OF THIS SCRIPT

grd= sprintf('EOGh,scale,Fz,legend,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nEMGl,O1,EMGf,O2,EMGr');
mnt= setDisplayMontage(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

grd= sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
mnt_lap= setDisplayMontage(mnt, grd);
mnt_spec= mnt_lap;
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


if ~isfield(opt,'band') | isempty(opt.band);
  bbci_bet_message('No band specified, select an optimal one: ');
  if isfield(opt,'ival') & ~isempty(opt.ival)
    opt.band = select_band(Cnt,mrk_selectClasses(mrk,bbci.classes),opt.ival);
  elseif isfield(opt,'default_ival')
    opt.band = select_band(Cnt,mrk_selectClasses(mrk,bbci.classes),opt.default_ival);
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
cnt_flt.title= sprintf('%s  [%d %d] Hz', Cnt.title, opt.band);
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
  set(gcf, fig_opt{:}, 'name',sprintf('%s: spectra in [%d %d] ms', Cnt.title, opt.ival));
  spec= proc_selectIval(epo, opt.ival);
  spec= proc_laplace(spec);
  spec= proc_spectrum(spec, [5 35], epo.fs);

  h = grid_plot(spec, mnt_spec, spec_opt);
  grid_markIval(opt.band);
  
  clear epo spec epo_rsq; 
end



if bbci.withgraphics
  bbci_bet_message('Creating figure 2\n');
  handlefigures('use','ERD');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: ERD in [%d %d] Hz', Cnt.title, opt.band));
  erd= makeEpochs(cnt_flt, mrk, opt.dar_ival);
  erd= proc_laplace(erd);
  erd= proc_rectifyChannels(erd);
  erd= proc_movingAverage(erd, 200);
  erd= proc_baseline(erd, [-500 0]);
  
  h = grid_plot(erd, mnt_lap, grid_opt);
  grid_markIval(opt.ival);

  clear erd epo_rsq;
end

bbci_bet_message('Outlierness\n');
fig1 = handlefigures('use','trial-outlierness');
fig2 = handlefigures('use','channel-outlierness');

mrk_cl= mrk_selectClasses(mrk, bbci.classes);
cnt_flt= proc_selectChannels(cnt_flt, opt.clab{:});
fv= makeEpochs(cnt_flt, mrk_cl, opt.ival);
if isfield(opt, 'threshold')
  fv = proc_outl_var(fv,struct('trialthresh',opt.threshold,...
                               'display',bbci.withclassification,...
                               'handles',[fig1,fig2]));
else
  proc_outl_var(fv,struct('display',bbci.withclassification,...
                          'handles',[fig1,fig2]));
end

bbci_bet_message('CSP\n');
switch opt.multicspmethod
 case 'ovr'
  [fv2, hlp_w, la]= proc_multicsp(fv, opt.nPat);
 case 'sim'
  [fv2, hlp_w, la]= proc_multicsp_sim(fv, opt.nPat);
end


clear cnt_flt

if bbci.withgraphics | bbci.withclassification,
  bbci_bet_message('Creating Figure CSP\n');
  switch opt.multicspmethod
   case 'ovr'
    [dum,ww,lam] =  proc_multicsp(fv, opt.nPat);
   case 'sim'
    [dum,ww,lam] = proc_multicsp_sim(fv, opt.nPat);
  end

  ww = pinv(ww)';
  lam = lam([1:opt.nPat,end-opt.nPat+1:end]);
  ww = ww(:,[1:length(bbci.classes)*opt.nPat,end-length(bbci.classes)*opt.nPat+1:end]);
  handlefigures('use','CSP');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: CSP-Pattern', Cnt.title));
  head= mnt_adaptMontage(mnt, fv.clab);
  plotCSPanalysis(fv, head, hlp_w, ww',la,'colAx','sym');
  fprintf('CSP outside xv: ');
end

if ~isfield(opt, 'usedPat')
  opt.usedPat = 1:2*opt.nPat;
end

opt.usedPat = intersect(opt.usedPat,1:2*opt.nPat);

bbci_bet_message('Calculate features\n');
features= proc_variance(fv2);
features= proc_logarithm(features);
features.x = features.x(:,opt.usedPat,:);
hlp_w = hlp_w(:,opt.usedPat);

if bbci.withclassification
  bbci_bet_message('outside classification: ');
  [loss,loss_std] = xvalidation(features, opt.model, [5 10]);
  remainmessage = sprintf('Outside classification: %2.1f +/- %1.1f\nInside Classification: ',100*loss,100*loss_std);
  bbci_bet_message('%2.1f +/- %1.1f\nInside Classification: ',100*loss,100*loss_std);
  switch opt.multicspmethod
   case 'ovr'
    fv.proc= ['fv= proc_multicsp(fv,' int2str(opt.nPat) '); ' ...
              'fv= proc_variance(fv); ' ...
              'fv= proc_logarithm(fv);'];
    %            'fv.x = fv.x(:,[', int2str(opt.usedPat),'],:);',...
   case 'sim'
    fv.proc= ['fv= proc_multicsp_sim(fv,' int2str(opt.nPat) '); ' ...
              'fv= proc_variance(fv); ' ...
              'fv= proc_logarithm(fv);'];
    %            'fv.x = fv.x(:,[', int2str(opt.usedPat),'],:);',...
  end
  [loss,loss_std] = xvalidation(fv, opt.model, [5 5]);
  bbci_bet_message('%2.1f +/- %1.1f\n',100*loss,100*loss_std);
  remainmessage = sprintf('%s%2.1f +/- %1.1f\n',remainmessage,100*loss,100*loss_std);
end


% What do we need later?
analyze = struct('csp_a', csp_a, 'csp_b', csp_b, 'csp_w', hlp_w, 'features', ...
                 features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');

