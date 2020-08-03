% bbci_bet_finish_adaptive script

% copy documentation from bbci_bet_finish_cspproto
global DATA_DIR
bbci.setup_opts = set_defaults(bbci.setup_opts, 'band', [4 35],...
								'ilen_apply', 10,...
								'clab',{'not','E*'},...
								'lap_clab',{'C3','C4','Cz'},...
								'aar_file',[DATA_DIR '/adaptive/aarfile.mat'],...
								'classy_file',[DATA_DIR '/adaptive/classyfile.mat'],...
								'offset',[1000 3000]);

bbci.setup_opts.clab = Cnt.clab(chanind(Cnt,bbci.setup_opts.clab));
Cnt1 = proc_selectChannels(Cnt,bbci.setup_opts.clab);

% Laplacian filter:
[cnt,A] = proc_laplace(Cnt1,getGrid('small'),'');
[d,a]= sort(cnt.clab);
[d,b] = sort(bbci.setup_opts.lap_clab);
ind = find(ismember(cnt.clab(a),bbci.setup_opts.lap_clab(b)));
[d,ind_reverse] = sort(b);
sel_ind = a(ind(ind_reverse));
% select the columns from A with the channels of interest
A = A(:,sel_ind);
% A will now project the data on  laplace-filtered signals with the channel labels 
% of interest (lap_clab).
cnt = proc_linearDerivation(Cnt1,A,'clab',bbci.setup_opts.lap_clab);
% cnt = proc_laplace(Cnt1,'small','');
% cnt = proc_selectChannels(cnt,bbci.setup_opts.lap_clab);
% Extract channel labels clab
analyze.clab = bbci.setup_opts.clab;
[b,a] = getButterFixedOrder(bbci.setup_opts.band, Cnt.fs, 5);

s_aar = load(bbci.setup_opts.aar_file);

cont_proc = strukt('clab',bbci.setup_opts.clab);
%cont_proc.procFunc = {'proc_laplace','proc_selectChannels','online_filt','proc_aar_online'};
%cont_proc.procParam = {{'small',''},{bbci.setup_opts.lap_clab},{b,a},{'state',s_aar.state}};
cont_proc.procFunc = {'proc_linearDerivation','online_filt','proc_aar_online'};
cont_proc.procParam = {{A,'clab',bbci.setup_opts.lap_clab},{b,a},{'state',s_aar.state}};
cont_proc.use_state_var = [0 1 0];

feature = struct('cnt',1);
feature.ilen_apply = bbci.setup_opts.ilen_apply;
feature.proc = {'proc_meanAcrossTime'};
feature.proc_param = {{''}};

cls = struct('fv',1);
cls.applyFcn = getApplyFuncName(opt.model);
s_cls = load(bbci.setup_opts.classy_file);
cls.C = s_cls.C;
cls.offset = bbci.setup_opts.offset;

% prepare the initialization of the proc_aar function:
bbci.initialize_functions = {'proc_aar_online'};
bbci.initialize_params = {{cnt,'state',s_aar.state,'init',true}};
