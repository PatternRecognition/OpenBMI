grd= sprintf('F9,scale,F3,Fz,F4,legend,F10\nFC7,FC5,FC3,FCz,FC4,FC6,FC8\nCP7,CP5,CP3,CPz,CP4,CP6,CP8\nP7,P3,P1,Pz,P2,P4,P8\nPO7,PO3,PO1,POz,PO2,PO4,PO8\nO5,O3,O1,Oz,O2,O4,O6');

% if isempty(chanind(Cnt,'EOGh')),
%   eog_bip= proc_bipolarChannels(Cnt,'EOGhp-EOGhn','EOGvp-EOGvn');
%   eog_bip.clab= {'EOGh','EOGv'};
%   Cnt= proc_selectChannels(Cnt,'not','EOG*');
%   Cnt= proc_appendChannels(Cnt,eog_bip);
% end

mnt= getElectrodePositions(Cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

fig_opt= {'numberTitle','off', 'menuBar','none'};

scalp_opt= struct('shading','flat', 'resolution',30, 'contour',[-40:2:40]);
head= setDisplayMontage(mnt, 'visible_128');
head= mnt_adaptMontage(head, Cnt.clab);
colDef= {'Target','Non-target';
         [0 0.7 0], [1 0 0]};
grid_opt = defopt_erps;
grid_opt.colorOrder = choose_colors(mrk.className,colDef);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};
scalp_opt.colorOrder = grid_opt.colorOrder;

handlefigures('use','ARTIFACT');
% [mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 1, 'do_multipass', 1, 'do_bandpass', 0, 'whiskerlength', 2.5)
% Cnt = proc_selectChannels(Cnt, 'not', rclab);
% epo = cntToEpo(Cnt,mrk_rej,opt.ival);

epo = cntToEpo(Cnt,mrk,opt.ival);

clear Cnt;
% epoMain = cntToEpo(Cnt,mrk,opt.ival);
epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
% epo = proc_setArtifactNoclass(epo, {'Fp2','F9'}, 70);
epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
opt.selectival = select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',2, 'visualize', 0, 'sort', 1, 'visu_scalps', 0, 'sign', 1);
opt.selectival = [opt.selectival select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',1, 'visualize', 0, 'sort', 1, 'visu_scalps', 0, 'sign', -1)];

% opt.selectival =[110 170; 210 240; 270 390];
% opt.selectival =[ 170   190;   230   280;   310   440];
% opt.selectival =[210 240; 270 350; 430 480];
% opt.selectival = [160 200; 240 310; 310 410]

features= proc_jumpingMeans(epo, opt.selectival);

%% There's a problem here with proc_flaten
[features, opt.meanOpt] = proc_subtractMean(proc_flaten(features));
[features, opt.normOpt] = proc_normalize(features);

if bbci.withgraphics
  handlefigures('use','INTERVAL');
  select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',3, 'visualize', 1, 'visu_scalps', 1);
%   set(gcf, fig_opt{:}, 'name',sprintf('%s: ERPs in [%d %d] ms', Cnt.title, opt.dar_ival));
  
  epo= proc_baseline(epo, opt.dar_base, 'beginning_exact');
  epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
  
  handlefigures('use','ERP');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: ERPs in [%d %d] ms', epo.title, opt.dar_ival));  
  h = grid_plot(epo, mnt, grid_opt);
  
  hh = cmap_posneg(81);
%   hh = hot(64); 
%   hh = hh(end:-1:1,:);
%   hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_markIval(opt.selectival);
  grid_addBars(epo_r,...
               'colormap',hh,'height',0.12,'h_scale',h.scale);
 
  handlefigures('use','ERPscalps');
  set(gcf, fig_opt{:}, ...
           'name',sprintf('%s: ERP-Pattern', epo.title));
  scalpEvolutionPlusChannel(epo, head, {'Pz', 'O2'}, opt.selectival, scalp_opt);
  grid_addBars(epo_r, 'colormap',hh,'height',0.12);

  handlefigures('use','ERP r-value scalps');  
  scalpEvolution(epo_r, head, opt.selectival, scalp_opt, 'colormap', cmap_posneg(81));
  clear epoMain epo_r 
end

if bbci.withclassification
  opt_xv= strukt('xTrials', [5 5], 'loss','rocArea');
  [dum,dum,outTe] = xvalidation(features, opt.model, opt_xv);
  me= val_confusionMatrix(features, outTe, 'mode','normalized');
  remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
  bbci_bet_message(remainmessage,0);
else
    remainmessage = '';
end

% What do we need later?
analyze = struct('features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
