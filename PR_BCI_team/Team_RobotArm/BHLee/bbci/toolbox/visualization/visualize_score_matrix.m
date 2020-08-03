function H= visualize_score_matrix(epo_r, ival, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'mnt', getElectrodePositions(epo_r.clab), ...
                  'opt_visu', [], ...
                  'visu_scalps', 1);

[opt_visu, isdefault]= ...
    set_defaults(opt.opt_visu, ...
                 'clf', 1, ...
                 'colormap', cmap_posneg(51), ...
                 'mark_clab', {'Fz','FCz','Cz','CPz','Pz','Oz'}, ...
                 'xunit', 'ms');
epo_r= proc_selectChannels(epo_r, scalpChannels); %% order channels
clf;
colormap(opt_visu.colormap);
if opt.visu_scalps,
  subplotxl(2, 1, 1, [0.05 0 0.01], [0.05 0 0.05]);
end
H.image= imagesc(epo_r.t, 1:length(epo_r.clab), epo_r.x'); 
H.ax= gca;
set(H.ax, 'CLim',[-1 1]*max(abs(epo_r.x(:)))); 
H.cb= colorbar;
cidx= strpatternmatch(opt_visu.mark_clab, epo_r.clab);
set(H.ax, 'YTick',cidx, 'YTickLabel',opt_visu.mark_clab, ...
          'TickLength',[0.005 0]);
if isdefault.xunit & isfield(epo_r, 'xUnit'),
  opt_visu.xunit= epo_r.xUnit;
end
xlabel(['[' opt_visu.xunit ']']);
ylabel('channels');
ylimits= get(H.ax, 'YLim');
set(H.ax, 'YLim',ylimits+[-2 2], 'NextPlot','add');
ylimits= ylimits+[-1 1];
for ii= 1:size(ival,1),
  xx= ival(ii,:);
  H.box(:,ii)= line(xx([1 2; 2 2; 2 1; 1 1]), ...
                    ylimits([1 1; 1 2; 2 2; 2 1]), ...
                    'color',[0 0.5 0], 'LineWidth',0.5);
end

%set(H.ax_overlay, 'Visible','off');
if opt.visu_scalps,
  nIvals= size(ival,1);
  for ii= 1:nIvals,
    H.ax_scalp(ii)= subplotxl(2, nIvals, nIvals + ii);
  end
%  ival= visutil_correctIvalsForDisplay(ival, 'fs',epo_r.fs);
  H.h_scalp= scalpEvolution(epo_r, opt.mnt, ival, defopt_scalp_r, ...
                            'subplot', H.ax_scalp, ...
                            'ival_color', [0 0 0], ...
                            'globalCLim', 1, ...
                            'scalePos','none');
  delete(H.h_scalp.text);
end
