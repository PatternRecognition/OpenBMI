% bbci_bet_analyze_cspproto script
% IN: subject -  string with the name of the subject.
%     date    -  date string, e.g. '06_10_30'
%     class_tag -  class tag like 'LR' or 'LF'
%    
% NOTE: this requires to generate an epo-struct first and store it under
% [DATA_DIR 'results/csp_paramspace_online/']. (see load_data_files.m in
% BCI_DIR/kraulems_analysis/csp_paramspace_online/)

% kraulem 10/06
bbci.setup_opts = set_defaults(bbci.setup_opts,'nPat',3);

%%%%%%%%%%%%%%%%%%%%%%%
% find the filters
%%%%%%%%%%%%%%%%%%%%%%%
s = load(bbci.epo_file); % alte vorverarbeitete epos
[fv1,csp_w1,csp_a1,csp_r1,theta] = proc_csp_prototypes(s.epo_st,3, ...
						  bbci.setup_opts.nPat);
[fv2,csp_w2,csp_eig,csp_a2] = proc_csp3(s.epo_st,bbci.setup_opts.nPat);
csp_w = [csp_w1 csp_w2];
fv = proc_linearDerivation(s.epo_st,csp_w);
fv = proc_variance(fv);
fv = proc_logarithm(fv);
mnt = setElectrodeMontage(s.epo_st.clab);

%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%
fig_opt = {};
bbci_bet_message('Creating Figure CSP\n');
handlefigures('use','CSPproto');
set(gcf, fig_opt{:},  ...
	 'name',sprintf('%s: CSP Proto <%s> vs <%s>', Cnt.short_title, ...
			bbci.classes{:}));
plotCSPanalysis(s.epo_st, mnt, csp_w1, csp_a1', 1:size(csp_a1,2));
bbci_bet_message('Creating Figure CSP\n');
handlefigures('use','CSPhist');
set(gcf, fig_opt{:},  ...
	 'name',sprintf('%s: CSP Hist <%s> vs <%s>', Cnt.short_title, ...
			bbci.classes{:}));
plotCSPanalysis(s.epo_st, mnt, csp_w2, csp_a2, 1:size(csp_a2,1));
analyze = struct;
analyze.features = fv;
analyze.csp_w = csp_w;
analyze.csp_a = s.a;
analyze.csp_b = s.b;

%%%%%%%%%%%%%%%%%%%%%%%
% store the filters.
%%%%%%%%%%%%%%%%%%%%%%%
% Extract channel labels clab
analyze.clab = Cnt.clab(chanind(Cnt,s.epo_st.clab));

