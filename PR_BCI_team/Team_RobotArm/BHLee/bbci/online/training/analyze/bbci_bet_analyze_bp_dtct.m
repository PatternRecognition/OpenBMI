%BBCI_BET_ANALYZE_BP_DTCT ANALYZES THE DATA PROVIDED BY
%BBCI_BET_PREPARE AND THE SPECIFIC SUBJECT FILE BY MEANS OF PLOTS
%AND CLASSIFICATION PERFORMANCES. FURTHERMORE IT PROVIDES THE
%FEATURES FOR BBCI_BET_FINISH_BP_DTCT. 
% THIS IS DONE FOR THE BEREITSCHAFTSPOTENTIAL FEATURE
% 
% input:
%   bbci   struct, the general setup variable from bbci_bet_prepare
%   opt    a struct with fields
%       clab    the used channels
%       ival    the train interval
%       band    the fft-filter band
%       fftparams parameters for the fft-filter
%       jMeans  number of samples to average
%       laplace  Apply laplace filter or not
%       dar_ival    the ival the plots are visualized
%       model   the model
%   Cnt, mrk, mnt
%
% output:
%   analyze  struct, will be passed on to 
%            bbci_bet_finish_bp_dtct
%
% dornheg, kraulem 07/05


% Everything that should be carried on to 
% bbci_bet_finish_bp_dtct must go
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


disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(Cnt, disp_clab);
mrk_mv = mrk_selectClasses(mrk,bbci.classes);
mrk_mv = mrk_setClasses(mrk,{1:length(mrk.pos)},{'move'});


if bbci.withgraphics
  bbci_bet_message('Creating figure 1\n');
  handlefigures('use','LRP');
  set(gcf, fig_opt{:},  ...
	   'name',sprintf('%s: LRP in [%d %d] Hz', Cnt.title, opt.band));
  cnt= proc_selectChannels(Cnt, opt.clab{:});
  
  % select epochs for movements and non-movements.
  epo_mv= makeEpochs(cnt, mrk_mv, opt.dar_ival);
  epo_rt= makeEpochs(cnt, mrk_mv, opt.dar_ival+opt.rest_shift);
  epo_rt.className = {'rest'};
 
  % attention: when appending the epochs, the field t is taken from epo_mv.
  epo = proc_appendEpochs(epo_mv,epo_rt);

  epo = proc_movingAverage(epo,50);
  epo= proc_baseline(epo, 300);
  epo_rsq= proc_r_square(epo);

  h = grid_plot(epo, mnt_lap, grid_opt);
  hh = hot(64); hh = hh(end:-1:1,:);
  hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_addBars(epo_rsq,'colormap',hh,'height',0.06,'h_scale',h.scale); 
  grid_markIval(opt.ival);

  clear epo epo_rsq;
end



bbci_bet_message('Calculate features\n');
%mrk_cl= mrk_selectClasses(mrk, bbci.classes);
cnt= proc_selectChannels(Cnt, opt.clab{:});
epo_mv= makeEpochs(cnt, mrk_mv, opt.ival);
epo_rt= makeEpochs(cnt, mrk_mv, opt.ival+opt.rest_shift);
epo_rt.className = {'rest'};
  
% attention: when appending the epochs, the field t is taken from epo_mv.
fv = proc_appendEpochs(epo_mv,epo_rt);
%fv= makeEpochs(cnt, mrk_cl, opt.ival);

if opt.laplace
  fv = proc_laplace(fv);
end
features = proc_selectChannels(fv,opt.classiclab{:});
features = proc_filtBruteFFT(features,opt.band,opt.fftparams{1},...
			     opt.fftparams{2});
if isfield(opt,'jMeans')&~isempty(opt.jMeans)
  features = proc_jumpingMeans(features,opt.jMeans);
end

% some outlierplots:
bbci_bet_message('Outlierness\n');
hfig = handlefigures('use','trial-outlierness');

if isfield(opt, 'threshold')
  features = proc_outl_slow(features,struct('trialthresh',opt.threshold,...
                               'display',bbci.withclassification,...
                               'handles',hfig));
else
  proc_outl_slow(features,struct('display',bbci.withclassification,...
                          'handles',hfig));
end

clear cnt
% Classify:
if bbci.withclassification
  [loss,loss_std] = xvalidation(features, opt.model, [5 5]);
  remainmessage = sprintf('Inside Classification: ',100*loss,100*loss_std);
  bbci_bet_message(remainmessage);
end


% What do we need later?
analyze = struct( 'features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
%
