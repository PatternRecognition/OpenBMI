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
%       band    the used band (for each rhythm one row)
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
% $Id: bbci_bet_analyze_csp2.m,v 1.10 2007/11/12 08:21:05 neuro_cvs Exp $


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
%analyze = struct;
analyze = [];

%TODO: DOCUMENTATION OF THIS SCRIPT

default_grd= ...
    sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'artifact_rejection', 0, ...
                 'channel_rejection', 1, ...
                 'outlier_rejection', 1, ...
                 'ival', [], ...
                 'default_ival', [1000 3000], ...
                 'repeat_bandselection', 1, ...
                 'usedPat', 1:2*opt.nPat, ...
                 'grd', default_grd, ...
                 'vis_laplace', 1, ...
                 'verbose', 1);
bbci_bet_memo_opt= ...
    set_defaults(bbci_bet_memo_opt, ...
                 'nPat', NaN, ...
                 'usedPat', NaN, ...
                 'band', NaN);
                                              
if isdefault.default_ival,
  msg= sprintf('default ival not define in bbci.setup_opts, using [%d %d]', ...
               opt.default_ival);
  warning(msg);
end

mnt= mnt_setGrid(mnt, opt.grd);

colOrder= [[1 0 0];[0 0.7 0];[0 0 1];[0 1 1];[1 0 1]; [1 1 0]];
grid_opt= struct('colorOrder', colOrder);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};

spec_opt= grid_opt;
spec_opt.yUnit= 'power';
spec_opt.xTickMode= 'auto';
spec_opt.xTick= 10:5:30;
spec_opt.xTickLabelMode= 'auto';
spec_opt.shrintAxes= [0.9 0.9];
rsqu_opt= {'colorOrder','rainbow'};

fig_opt= {'numberTitle','off', 'menuBar','none'};

%% when nPat was changed, but usedPat was not, define usedPat
if bbci_bet_memo_opt.nPat~=opt.nPat ...
      & isequal(bbci_bet_memo_opt.usedPat, opt.usedPat),
  opt.usedPat= 1:2*opt.nPat;
end

if ~isequal(bbci.classes, 'auto'),
  mrk= mrk_selectClasses(mrk, bbci.classes);
  if opt.verbose,
    fprintf('using classes <%s> as specified\n', vec2str(bbci.classes));
  end
end

clab= opt.clab;

%% artifact rejection (trials and/or channels)
if opt.artifact_rejection | opt.channel_rejection,
  if opt.verbose,
    fprintf('checking for artifacts\n');
  end
  cnt_tmp= proc_selectChannels(Cnt, clab);
  [mk_clean , rClab, rTrials]= ...
      reject_varEventsAndChannels(cnt_tmp, mrk, [500 4500]);
  if opt.artifact_rejection,
    if length(rTrials)>0 | opt.verbose,
      fprintf('rejected: %d trials.\n', length(rTrials));
    end
    mrk= mk_clean;
  end
  if opt.channel_rejection,
    if length(rClab)>0 | opt.verbose,
      fprintf('rejected channels: <%s>.\n', vec2str(rClab));
    end
    clab= setdiff(cnt_tmp.clab, rClab);
    Cnt= proc_selectChannels(Cnt, 'not', rClab);
  end
  clear cnt_tmp
end

clear analyze
nClassesConsidered= size(mrk.y,1);
class_combination= nchoosek(1:nClassesConsidered, 2);
for ci= 1:size(class_combination,1),
  
classes= mrk.className(class_combination(ci,:))
if nClassesConsidered>2,
  fprintf('investigating class combination <%s> vs <%s>\n', ...
          classes{:});
end

band_fresh_selected= 0;
if isempty(opt.band);
  bbci_bet_message('No band specified, select an optimal one: ');
  if ~isempty(opt.ival)
    ival_for_bandsel= opt.ival;
  else
    ival_for_bandsel= opt.default_ival;
    if ~opt.repeat_bandselection,
      bbci_bet_message('You should run bbci_bet_analyze a 2nd time.');
    end
  end
  opt.band= select_bandnarrow(Cnt, mrk2, ival_for_bandsel);
  bbci_bet_message('[%g %g] Hz\n',round(opt.band));
  band_fresh_selected= 1;
end

if ~isequal(opt.band, bbci_bet_memo_opt.band),
  [filt_b,filt_a]= butters(5, opt.band/Cnt.fs*2);
  clear cnt_flt
  cnt_flt= proc_filterbank(Cnt, filt_b, filt_a);
  if opt.verbose,
    bbci_bet_message('Data filtered\n');  
  end
elseif opt.verbose,
  bbci_bet_message('Filtered data reused\n');    
end

if isempty(opt.ival),
  bbci_bet_message('No ival specified, automatic selection: ');
  tmp_ival = zeros(size(opt.band,1),2);
  for i= 1:size(opt.band,1),
    cnt_tmp= proc_selectChannels(cnt_flt, ['*flt' int2str(i)]);
    for j= 1:length(cnt_tmp.clab)
      cnt_tmp.clab{j}(end-4-floor(log10(i)):end) = '';
    end
    tmp_ival(i,:)= select_ival(cnt_tmp, mrk2);
    clear cnt_tmp
  end
  %% BB: this is definitly not an intelligent solution...
  opt.ival= mean(tmp_ival, 1);
  bbci_bet_message('[%i %i] msec.\n', opt.ival);
elseif size(opt.ival,1) > 1,
  error('Only *one* ival may be given.');
end

if opt.repeat_bandselection & band_fresh_selected & ...
      ~isequal(opt.ival, ival_for_bandsel),
  bbci_bet_message('Redoing selection of frequency band for new interval: ');
  opt.band= select_bandnarrow(Cnt, mrk2, opt.ival);
  bbci_bet_message('[%g %g] Hz\n',round(opt.band));
end

disp_clab= getClabOfGrid(mnt);
if opt.vis_laplace,
  requ_clab= getClabForLaplace(Cnt, disp_clab);
else
  requ_clab= disp_clab;
end

if bbci.withgraphics 
  epo= proc_selectChannels(Cnt, requ_clab);
  epo= makeEpochs(epo, mrk, opt.dar_ival);
  if opt.verbose>=2,
    bbci_bet_message('Creating figure for spectra\n');
  end
  handlefigures('use','Spectra',1);
  set(gcf, fig_opt{:}, 'name',...
           sprintf('%s: spectra in [%d %d] ms', Cnt.short_title, opt.ival));
  spec= proc_selectIval(epo, opt.ival);
  spec= proc_laplace(spec);
  spec= proc_spectrum(spec, [5 35], kaiser(epo.fs,2), epo.fs/4);
  spec_rsq= proc_r_square(proc_selectClasses(spec,classes));
    
  h = grid_plot(spec, mnt, spec_opt);
  grid_markIval(opt.band);
  grid_addBars(spec_rsq,'colormap',flipud(hot(64)), ...
               'height',0.12, 'h_scale',h.scale, 'cLim','0tomax');
    
  clear epo spec_rsq spec
  handlefigures('next_fig','Spectra');
end


if bbci.withgraphics
  if opt.verbose>=2,
    bbci_bet_message('Creating figure(s) for ERD\n');
  end
  handlefigures('use','ERD',size(opt.band,1));
  for i = 1:size(opt.band,1),
    set(gcf, fig_opt{:},  ...
             'name',sprintf('%s: ERD for [%g %g] Hz', ...
                            Cnt.short_title, opt.band(i,:)));
    erd = proc_selectChannels(cnt_flt, ['*flt' int2str(i)]);
    for j = 1:length(erd.clab)
      erd.clab{j}(end-4-floor(log10(i)):end) = '';
    end
    erd= proc_laplace(erd);
    erd= proc_envelope(erd, 'ms_msec', 200);
    erd= proc_baseline(erd, [], 'trialwise',0);
    erd= makeEpochs(erd, mrk, opt.dar_ival);
    epo_rsq= proc_r_square(proc_selectClasses(erd, classes));
  
    h = grid_plot(erd, mnt, grid_opt);
    grid_markIval(opt.ival);
    grid_addBars(epo_rsq,'colormap',flipud(hot(64)), ...
                 'height',0.06, 'h_scale',h.scale, 'cLim','0tomax'); 
    
    clear erd epo_rsq;
    handlefigures('next_fig','ERD');
  end
  
end

for i= 1:size(opt.band,1),
  ff= proc_selectChannels(cnt_flt, ['*flt' int2str(i)]);
  for j= 1:length(ff.clab),
    ff.clab{j}(end-4-floor(log10(i)):end) = '';
  end
  ff= proc_selectChannels(ff, clab);
  ff= makeEpochs(ff, mrk2, opt.ival);
  ff.clab= apply_cellwise(ff.clab, 'strcat', ['_flt' int2str(i)]);
  if i == 1
    fv= ff;
  else
    fv= proc_appendChannels(fv, ff);
  end
end


warning('Matthias has to do some outlier stuff');
if opt.outlier_rejection,
  if opt.verbose,
    bbci_bet_message('checking for outliers\n');
  end
  fig1 = handlefigures('use','trial-outlierness',size(opt.band,1));
  fig2 = handlefigures('use','channel-outlierness',size(opt.band,1));
  %% BB: I propose to do outlier removal on broadband signals,
  %%  NOT separately for each band.
  for i = 1:size(opt.band,1)
    %if isfield(opt, 'threshold') & opt.threshold<inf
    %  fv = proc_outl_var(fv, 'trialthresh',opt.threshold,...
    %		     'display',bbci.withclassification,...
    %		     'handles',[fig1,fig2]);
    %else
    proc_outl_var(proc_selectChannels(fv, ['*flt1']), ...
                  'display', bbci.withclassification,...
                  'handles', [fig1(i),fig2(i)]);
    handlefigures('next_fig', 'trial-outlierness');
    handlefigures('next_fig', 'channel-outlierness');
  end
end

if opt.verbose,
  bbci_bet_message('calculating CSP\n');
end

%% Vorsicht, Hacker unterwegs
%% make hlp_w global such that it can be accessed from xvalidation
%% in order to match patterns with these ones
global hlp_w
hlp_w= zeros(length(fv.clab),2*opt.nPat);
la= zeros(size(opt.band,1)*opt.nPat*2,1);
A= zeros(2*opt.nPat,size(opt.band,1));

posi = 0;
for i= 1:size(opt.band,1),
  ff= proc_selectChannels(fv,['*flt' int2str(i)]);
  nff= length(ff.clab);
  idx= (i-1)*2*opt.nPat+1:i*2*opt.nPat;
  [ff, hlp_w(posi+1:nff+posi, idx), la(idx,1), A(idx,posi+1:nff+posi)]= ...
      proc_csp3(ff, 'patterns',opt.nPat, 'scaling','maxto1');
  posi = posi+nff;
  ff.clab= apply_cellwise(ff.clab, 'strcat', ['_flt' int2str(i)]);
  if i == 1
    f= ff;
  else
    f= proc_appendChannels(f,ff);
  end
end
fv2= f;


if ~iscell(opt.usedPat)
  opt.usedPat = {opt.usedPat};
end
if length(opt.usedPat)==1
  opt.usedPat = repmat(opt.usedPat,[1,size(opt.band,1)]);
end
if length(opt.usedPat)~=size(opt.band,1)
  warning('mismatching fields usedPat');
  opt.usedPat = opt.usedPat{1};
end

usedPat = [];
for i= 1:length(opt.usedPat),
  usedPat= [usedPat, (i-1)*opt.nPat*2+opt.usedPat{i}];
end


if bbci.withgraphics | bbci.withclassification,
  posi= 0;
  bbci_bet_message('Creating Figure(s) of CSP analysis\n');
  handlefigures('use', 'CSP',size(opt.band,1));
  for i= 1:size(opt.band,1),
    set(gcf, fig_opt{:},  ...
             'name',sprintf('%s: CSP <%s> vs <%s>, [%g %g] Hz', ...
                            Cnt.short_title, classes{:},opt.band(i,:)));
    cnt_tmp= proc_selectChannels(fv, ['*flt' int2str(i)]);
    for j= 1:length(cnt_tmp.clab),
      cnt_tmp.clab{j}(end-4-floor(log10(i)):end)= '';
    end
    nff= length(cnt_tmp.clab);
    idx= (i-1)*2*opt.nPat+1:i*2*opt.nPat;
    plotCSPanalysis(cnt_tmp, mnt, hlp_w(posi+1:nff+posi,idx),...
                    A(idx,posi+1:nff+posi), la(idx,1), ...
                    'mark_patterns', opt.usedPat{i});
    
    posi= posi+nff;
    handlefigures('next_fig','CSP');
    clear cnt_tmp
  end
end


if opt.verbose,
  bbci_bet_message('calculate features\n');
end

features= proc_variance(fv2);
features= proc_logarithm(features);
features.x= features.x(:,usedPat,:);
hlp_w= hlp_w(:,usedPat);


%% BB: I propose a different validation. For test samples always take a
%%  FIXED interval, e.g. opt.default_ival. Practically this can be done
%%  with bidx, train_jits, test_jits.
if bbci.withclassification,
  opt_xv= strukt('sample_fcn',{'chronKfold',8}, 'std_of_means',0);
  [loss,loss_std]= xvalidation(features, opt.model, opt_xv);
  bbci_bet_message('CSP global: %4.1f +/- %3.1f\n', 100*loss, 100*loss_std);
  remainmessage= sprintf('CSP global: %4.1f +/- %3.1f', 100*loss, 100*loss_std);
  proc= struct('memo', 'csp_w');
  proc.train= ['ff= proc_selectChannels(fv, ''*flt1''); ' ...
               '[ff,csp_w]= proc_csp3(ff, ''patterns'',' int2str(opt.nPat) ');'];
  for i= 2:size(opt.band,1),
    proc.train= [proc.train, ...
                 'fff = proc_selectChannels(fv, ''*flt' int2str(i) '''); ' ...
                 '[fff,csp_ww]= proc_csp3(fff, ''patterns'',' int2str(opt.nPat) ');' ...
                 'ff= proc_appendChannels(ff, fff); ' ...
                 'csp_w= [csp_w, zeros(size(csp_w,1),size(csp_ww,2)); ' ...
                         'zeros(size(csp_ww,1),size(csp_w,2)), csp_ww];'];
  end
  proc.train= [proc.train, ...
               'fv= proc_variance(ff); ' ...
               'fv= proc_logarithm(fv);'];
  proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  

  [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc',proc);
  bbci_bet_message('CSP inside: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
  remainmessage= sprintf('%s\nCSP inside: %4.1f +/- %3.1f', ...
                         remainmessage, 100*loss,100*loss_std);
  flag= true;
  for i= 1:size(opt.band,1)
    flag= flag*isequal(opt.usedPat{i}, 1:2*opt.nPat);
  end
  if ~flag,
    proc.train = ['global hlp_w; ' ...
                  'hlp_ind= chanind(fv.clab,''*flt1''); ' ...
                  'ff= proc_selectChannels(fv, hlp_ind); ' ...
                  '[ff,csp_w]= proc_csp3(ff, ''patterns'',hlp_w(hlp_ind,find(any(hlp_w(hlp_ind,:),1))),''selectPolicy'',''matchfilters'');'];

    for i= 2:size(opt.band,1),
      proc.train = [proc.train, ...
                    'hlp_ind = chanind(fv.clab,''*flt' int2str(i) '''); ' ...
                    'fff = proc_selectChannels(fv, hlp_ind); ' ...
                    '[fff,csp_ww]= proc_csp3(fff, ''patterns'',hlp_w(hlp_ind,find(any(hlp_w(hlp_ind,:),1))),''selectPolicy'',''matchfilters''); ' ...
                    'ff= proc_appendChannels(ff, fff); ' ...
                    'csp_w = [csp_w, zeros(size(csp_w,1),size(csp_ww,2)); ' ...
                             'zeros(size(csp_ww,1),size(csp_w,2)), csp_ww];'];
    end
    proc.train= [proc.train, ...
                 'fv= proc_variance(ff); ' ...
                 'fv= proc_logarithm(fv);'];
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv);'];
    
    [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc',proc);
    bbci_bet_message('CSP selPat: %4.1f +/- %3.1f\n', 100*loss, 100*loss_std);
    remainmessage= sprintf('%s\nCSP setPat: %4.1f +/- %3.1f', ...
                           remainmessage, 100*loss, 100*loss_std);
  end
  mean_loss(ci)= loss;
  std_loss(ci)= loss_std;
end

end  %% for ci 

if isequal(bbci.classes, 'auto'),
  [dmy, bi]= min(mean_loss + 0.1*std_loss);
  bbci.classes= mrk.className(class_combination(bi,:));
  bbci.class_selection_loss= [mean_loss; std_loss];
end

% Gather all information that should be saved in the classifier file
analyze(ci)= struct('csp_a', {filt_a}, ...
                    'csp_b', {filt_b}, ...
                    'csp_w', {hlp_w}, ...
                    'features', {features}, ...
                    'message', {remainmessage});

if opt.verbose,
  bbci_bet_message('Finished analysis\n');
end
