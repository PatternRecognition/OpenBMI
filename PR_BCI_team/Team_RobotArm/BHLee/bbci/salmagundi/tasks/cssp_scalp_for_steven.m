
dbstop inclassDef= {1, 2, 3;
           'left', 'right', 'foot'};

BC.folder= EEG_RAW_DIR;
BC.file= 'VPkg_08_08_07/imag_arrowVPkg';
BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {classDef};
BC.save.folder= TMP_DIR;
BC.save.file= 'bbci_classifier_CSP';
BC.analyze_fcn= @bbci_calibrate_CSP;

bbci= [];
bbci.calibrate= BC;

dbstop in bbci_calibrate_CSP at 343
[bbci, data]= bbci_calibrate(bbci);

tau= 70;
fv_tau= proc_addDelayedChannels(fv, tau);
[fv2, hlp_w, la, A]= proc_csp3(fv_tau, 'patterns',opt.nPatterns);

sz= size(hlp_w);
W= reshape(hlp_w, [sz(1)/2 sz(2)*2]);
A= reshape(A, [sz(2)*2 sz(1)/2]);
fig_set(2); clf;
opt_scalp_csp= strukt('colormap', cmap_greenwhitelila(31));
plotCSPanalysis(fv, mnt, W, A, [la'; la']', opt_scalp_csp);

nice_map= W(:,2);
fig_set(1); clf;
mt= mnt_adaptMontage(mnt, fv.clab);
cmap= cmap_rainbow(21);
opt_scalp= defopt_scalp_r2('contour', 0, ...
                           'scalePos','none', ...
                           'showLabels', 0, ...
                           'lineProps', {'linewidth', 5}, ...
                           'ears', 1, ...
                           'colormap', cmap);
scalpPlot(mt, nice_map, opt_scalp);

w2= sign(nice_map).*sqrt(abs(nice_map));
H= scalpPlot(mt, w2, opt_scalp);
delete(H.label_markers);
printFigure('/tmp/cssp_map', [10 10], 'format','pdf');
