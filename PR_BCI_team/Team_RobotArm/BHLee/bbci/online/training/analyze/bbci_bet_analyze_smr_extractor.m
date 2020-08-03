% blanker@cs.tu-berlin.de, Feb-2010

% Everything that should be carried over to bbci_bet_finish_sellap must go
% into the variable analyze.

% get access to proc_peakAreaFit
path(path, [BCI_DIR 'investigation/studies/vitalbci_season1']);
path(path, [BCI_DIR 'acquisition\setups\smr_neurofeedback_season1']);

%% Define default values for optinal properties
if ismember('CCP1', Cnt.clab),
  default_grd= ...
      sprintf('scale,FC3,FC1,FCz,FC2,FC4,legend\n<,CFC5,CFC3,CFC1,CFC2,CFC4,CFC6\nC5,C3,C1,Cz,C2,C4,C6\n<,CCP5,CCP3,CCP1,CCP2,CCP4,CCP6\nCP5,CP3,CP1,CPz,CP2,CP4,CP6');
else
  default_grd= ...
      sprintf('scale,FC3,FC1,FCz,FC2,FC4,legend\nC5,C3,C1,Cz,C2,C4,C6\nCP5,CP3,CP1,CPz,CP2,CP4,CP6');
end

%% define the motor area where we should look for the peaks
%area_LH= {'FC5-3','CFC5-3','C5-3','CCP5-3','CP5-3'};
%area_C= {'FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2'};
%area_C= {'FCz','CFC1-2','Cz','CCP1-2','CPz'};
% restricted motor area withou central elecs
% area_LH= {'CCP3'};
% area_RH= {'CCP4'};

area_LH= {'FC3','CFC5-3','C3','CCP5-3','CP3'};
area_RH= {'FC4','CFC4-6','C4','CCP4-6','CP4'};
% alternative with more central elecs
%area_C= {'FC1-2','C1-2','CP1-2'};
% area_LH= {'FC1,3','CFC5-3','C1,3','CCP5-3','CP1,3'};
% area_RH= {'FC2,4','CFC4-6','C2,4','CCP4-6','CP2,4'};
%default_motorarea= {area_LH, area_C, area_RH};
default_motorarea= {area_LH, area_RH};
%%

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'clab', '*', ...
                 'reject_artifacts', 1, ...
                 'reject_channels', 1, ...
                 'reject_opts', {}, ...
                 'band', 'auto', ...
                 'band_search', 'alpha+beta', ...
                 'upper_band', [], ...
                 'extractor_fcn', 'smr_extractor', ...
                 'visu_band', [5 35], ...
                 'visu_specmaps', 1, ...
                 'grd', default_grd, ...
                 'motorarea', default_motorarea, ...
                 'spat_flt','lap');

mrk= mrkodef_artifacts(mrk_orig);
blk= blk_segmentsFromMarkers(mrk, ...
                             'start_marker','eyes open', ...
                             'end_marker','stop');
blk.className={'eyes open'};
blk.y= ones(1, size(blk.ival, 2)); 

[cnt, blkcnt]= proc_concatBlocks(Cnt, blk);
mrk= mrk_evenlyInBlocks(blkcnt, 1000);

%% Analysis starts here
clear fv*
clab= cnt.clab(chanind(cnt, opt.clab));

%% artifact rejection (trials and/or channels)
flds= {'reject_artifacts', 'reject_channels', ...
       'reject_opts', 'clab'};
if bbci_memo.data_reloaded | ...
      ~fieldsareequal(bbci_bet_memo_opt, opt, flds),
  clear anal
  anal.rej_trials= NaN;
  anal.rej_clab= NaN;
  if opt.reject_artifacts | opt.reject_channels,
    handlefigures('use', 'Artifact rejection', 1);
    set(gcf, 'Visible','off', ...
             'name',sprintf('%s: Artifact rejection', cnt.short_title));
    [mk_clean , rClab, rTrials]= ...
        reject_varEventsAndChannels(cnt, mrk, [0 1000], ...
                                    'clab',clab, ...
                                    'do_multipass', 1, ...
                                    opt.reject_opts{:}, ...
                                    'visualize', bbci.withgraphics);
    set(gcf,  'Visible','on');
    if opt.reject_artifacts,
      if not(isempty(rTrials)),
        %% TODO: make output class-wise
        fprintf('rejected: %d trial(s).\n', length(rTrials));
      end
      anal.rej_trials= rTrials;
    end
    if opt.reject_channels,
      if not(isempty(rClab)),
        fprintf('rejected channels: <%s>.\n', vec2str(rClab));
      end
      anal.rej_clab= rClab;
    end
  end
end
if iscell(anal.rej_clab),   %% that means anal.rej_clab is not NaN
  clab(strpatternmatch(anal.rej_clab, clab))= [];
end
if isequal(opt.spat_flt, 'lap')
  epo = proc_laplacian(cnt);
elseif isequal(opt.spat_flt, 'lar')
  epo = proc_localAverageReference(cnt,mnt,'radius',0.4);
else
  error('unvalid spatial filtering option!')
end

epo= cntToEpo(epo, mrk, [0 1000], 'mtsp','before');
spec= proc_spectrum(epo, [2 35], kaiser(epo.fs,2));
spec= proc_average(spec);

if isempty(opt.band) | isequal(opt.band, 'auto'),
  switch(opt.band_search),
    case 'alpha'
     apa = proc_peakSearch(spec, 6:11, 12:18);
    case 'beta'
     apa = proc_peakSearch(spec, 15:25, 20:35);
    case 'alpha+beta'
     apa = proc_peakSearch(spec, 6:11, 12:18);
     apa_beta = proc_peakSearch(spec, 15:25, 20:35);
     idx= find(apa_beta.x > apa.x);
     apa.x(idx)= apa_beta.x(idx);
     apa.peak_time(idx)= apa_beta.peak_time(idx);
     apa.ival(idx,:)= apa_beta.ival(idx,:);
    otherwise
      error('unknown option for bandsearch');
  end
else
  lower_bound= opt.band(1)+[-4:0];
  upper_bound= opt.band(2)+[0:4];
%  upper_bound= opt.band(1)+2:opt.band(2)+4;
  apa = proc_peakSearch(spec, lower_bound, upper_bound, 'intersect_with_band',opt.band);
end

mnt= mnt_setGrid(mnt, opt.grd);
opt_scalp= ...
    defopt_scalp_power('mark_properties',{'MarkerSize',7,'LineWidth',2});

handlefigures('use', 'Spectra', 1);
set(gcf, 'Visible','off', ...
         'name',sprintf('%s: Spectra', cnt.short_title));
H= grid_plot(spec, mnt, 'xTick',[10 20 30], 'titleDir','none', ...
             'shrinkAxes',[0.95 0.9], 'xTickAxes','CPz');
clab_list= intersect(getClabOfGrid(mnt), strhead(spec.clab));

% compute the highest peak in the specific given band
for ii= 1:length(clab_list),
  h= grid_getSubplots(clab_list{ii});
  ia= chanind(apa, clab_list{ii});
  axes(h); hold on;
  if isfield(apa, 'interpolation_ival'),
    ival= apa.interpolation_ival(ia,:);
  else
    ival= apa.ival(ia,:);
  end
  iv= getIvalIndices(ival, spec);
  lin= linspace(spec.x(iv(1),ia), spec.x(iv(end),ia), length(iv));
  plot(linspace(ival(1), ival(2), length(lin)), lin, 'r');
  %  ht= text(mean(ival), max(spec.x(iv,ia)), sprintf('%.1f', apa.x(ia)));
  ht= text(apa.peak_time(ia), max(spec.x(iv,ia)), sprintf('%.1f', apa.x(ia)));
  set(ht, 'horizontalAli','center', 'fontWeight','bold');
end

% select one electrode per MA witrh the highest peak
nSel= length(opt.motorarea);
sel_clab= cell(1, nSel);
for ii= 1:nSel,
  ic= chanind(apa, opt.motorarea{ii});
  [mm,mi]= max(apa.x(ic));
  sel_clab{ii}= apa.clab{ic(mi)};
end

% computes the central and side bands if it is not given
if isempty(opt.band) | isequal(opt.band, 'auto'),
  ci= chanind(apa, sel_clab);
  lower= apa.ival(ci, 1);
  opt.lower_band= [-2 0] + min(lower(:));
  upper= apa.ival(ci, 2);
  opt.upper_band= [0 3] + max(upper(:));
  peaks= apa.peak_time(ci);
  opt.band= [-2 2] + [min(peaks) max(peaks)];
  %% make central band disjoint from outer bands:
  opt.band= [max(opt.band(1),opt.lower_band(2)), min(opt.band(2),opt.upper_band(1))];
elseif ~isempty(opt.band) & isempty(opt.upper_band)
  %compute the upper and lower bands if they are not given beforehand
  ci= chanind(apa, sel_clab);
  opt.lower_band = min(apa.interpolation_ival(ci,1)) - [2 0];  
  opt.upper_band = max(apa.interpolation_ival(ci,2)) + [0 3];
end
%%
% mark the computed ivals in the grid_plot
grid_markIval(opt.band, sel_clab, [0.5 1 0.5]);
grid_markIval(opt.lower_band, sel_clab, [1 0.5 0.5]);
grid_markIval(opt.upper_band, sel_clab, [1 0.5 0.5]);
highlight= chanind({H.chan.clab}, sel_clab);
if ~isempty(highlight),
  set([H.chan(highlight).ax_title], 'FontWeight','bold', 'FontSize',12);
end
set(gcf, 'Visible','on');

if opt.visu_specmaps,
  handlefigures('use', 'Maps of SMR Peak', 1);
  set(gcf, 'name',sprintf('%s: Maps of SMR Peak', cnt.short_title));
  H= scalpPatterns(apa, mnt, [], opt_scalp, ... 
                   'mark_channels', sel_clab);  
end

% gather all information needed for online processing
bands= cat(1, opt.band, opt.lower_band, opt.upper_band);
[filt_b,filt_a]= butters(opt.filtOrder, bands/cnt.fs*2);

if isequal(opt.spat_flt, 'lap') 
  requ_clab= getClabForLaplacian(cnt, sel_clab);
  [cnt_flt, lap_w]= proc_laplacian(proc_selectChannels(cnt,requ_clab));
elseif isequal(opt.spat_flt, 'lar')
  requ_clab= getClabForLAR(cnt, mnt, sel_clab,0.4);
  [cnt_flt, lap_w]=proc_localAverageReference(proc_selectChannels(cnt,requ_clab), mnt, 'clab',sel_clab, 'radius',0.4);
else
  error('unvalid spatial filtering option!')
end
cnt_flt= proc_filterbank(cnt_flt, filt_b, filt_a);
mrk_ilen= mrk_evenlyInBlocks(blkcnt, opt.ilen_apply(1)/2);
fv= cntToEpo(cnt_flt, mrk_ilen, [0 opt.ilen_apply(1)], 'mtsp','before');
fv= proc_variance(fv);
fv= proc_logarithm(fv);
%[fv, opt_smr]= proc_smr_extractor(fv);
[fv, opt_smr]= feval(['proc_' opt.extractor_fcn], fv);
ff= cntToEpo(cnt_flt, mrk_ilen, [0 opt.ilen_apply(2)], 'mtsp','before');
ff= proc_variance(ff);
ff= proc_logarithm(ff);
%ff= proc_smr_extractor_adaptCentralBand(ff, opt_smr);
ff= feval(['proc_' opt.extractor_fcn], ff, opt_smr);

handlefigures('use', 'Distribution of SMR Values', 1);
set(gcf, 'name',sprintf('%s: Distribution of SMR Values', cnt.short_title));
plot(fv.x, '.');
set(gca, 'XLim',[0 length(fv.x)+1], 'YLim',[-0.2 1.2]);
hold on
perc = percentiles(fv.x, [15 85]);
plot([1 size(fv.x,2)], [perc(1) perc(1)], 'r--');
plot([1 size(fv.x,2)], [perc(2) perc(2)], 'r--');
plot(ff.x, 'k.');
legend(cprintf('%d ms', opt.ilen_apply));
hold off;

analyze= strukt('clab', requ_clab, ...
                'sel_lapclab', sel_clab, ...
                'filt_a', filt_a, ...
                'filt_b', filt_b, ...
                'spat_w', lap_w, ...
                'features', fv, ...
                'extractor_fcn', opt.extractor_fcn, ...
                'opt_smr', opt_smr, ...
                'opt_bands', [opt.band; opt.lower_band; opt.upper_band]);

bbci_memo.data_reloaded= 0;
%warning(wstate.state, 'bbci:multiple_channels');
