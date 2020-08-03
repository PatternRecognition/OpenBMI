function train_common_classifier = train_common_classifier(varargin)

global EEG_MAT_DIR DATA_DIR BCI_DIR

opt = set_defaults(opt, 'meth', 'cspp', ...
  'ipatch', 11, ...
  'icenter', 2, ...
  'StudyName', 'vitalbci_season1', ...
  'nPatPerPatch', 1, ...
  'csp_selectPolicy', 'equalperclass', ...
  'csp_score', 'medianvar', ...
  'nCSP', 3, ...
  'nPat', 3, ...
  'patch_selectPolicy', 'equalperclass', ...
  'patch_score', 'medianvar', ...
  'nLap', 6, ...
  'lap_selectPolicy', 'equalperclass', ...
  'lap_score', 'medianvar', ...
  'covPolicy', 'average', ...
  'require_complete_neighborhood', 1, ...
  'band', [8 32]);

%% check on the consistency of the options
if isequal(opt.meth, 'cspp') && ~isequal(opt.patch_selectPolicy, 'auto') && ~isequal(opt.patch_selectPolicy, 'all')
  if isequal(opt.patch_selectPolicy, 'equalperclass') || isequal(opt.patch_selectPolicy, 'directorscut')
    %     if opt.nPat == 6
    %       warning('with this policy option nPat per class will be calculated -> put opt.nPat = 3')
    %       opt.nPat = 3;
    %     end
  else
    if opt.nPat == 3
      warning('with this policy option nPat in total will be calculated -> put opt.nPat = 3')
      opt.nPat = 6;
    end
  end

  if ~isempty(findstr(opt.patch_selectPolicy, 'PerArea')) && opt.nPat == 3
    warning('with this policy option nPat/3 patterns per area will be chosen -> put opt.nPat = 6')
    opt.nPat = 6;
  end
end
if isequal(opt.meth, 'lap')
  if ~isempty(findstr(opt.lap_selectPolicy, 'PerArea')) && opt.nLap == 3
    warning('with this policy option nLap/3 laplacian per area will be chosen -> put opt.nPat = 6')
    opt.nLap = 6;
  end
end
if ~ischar(opt.covPolicy) && (~isequal(opt.csp_score, 'eigenvalues') || ~isequal(opt.patch_score, 'eigenvalues'))
  warning('using the class covariance matrix for csp and pacthes calculation allows to use just eigenvalues as score -> put opt.csp_score = eigenvalues and opt.patch_scroe = eigenvalues');
  opt.csp_score = 'eigenvalues';
  opt.patch_score = 'eigenvalues';
end

center_list_str = {'c12','C3z4', '5c', '3c3cp', '5c3cp', '3fc3c3cp', '3fc5c3cp', '3fc5c5cp', '5fc5c5cp', ...
  '5fc5c5cp3p','5fc5c5cp1p','c1z2','2c34','4c34','2c342cp34','2fc342c342cp34'};
center_list = {{'C1,2'},{'C3,z,4'}, {'C3,1,z,2,4'}, {'C3,z,4', 'CP3,z,4'}, {'C3,1,z,2,4', 'CP3,z,4'}, ...
  {'FC3,z,4', 'C3,z,4', 'CP3,z,4'}, {'FC3,z,4', 'C3,1,z,2,4', 'CP3,z,4'}, ...
  {'FC3,z,4', 'C3,1,z,2,4', 'CP3,1,z,2,4'}, {'FC3,1,z,2,4', 'C3,1,z,2,4', 'CP3,1,z,2,4'}, ...
  {'FC3,1,z,2,4', 'C3,1,z,2,4', 'CP3,1,z,2,4', 'P3,z,4'}, {'FC3,1,z,2,4', 'C3,1,z,2,4', 'CP3,1,z,2,4', 'Pz'},{'C1,z,2'},{'C3,4'},{'C3,1,2,4'}, {'C3,4', 'CP3,4'},{'FC3,4', 'C3,4', 'CP3,4'}};
patch_list = {'small', 'sixnew', 'six', 'large', 'eightnew','eight','eightsparse', 'ten','eleven','eleven_to_anterior','twelve','eighteen','twentytwo'};

clab64 = {'F5,3,1,z,2,4,6','FFC5,3,1,z,2,4,6','FC5,3,1,z,2,4,6','CFC5,3,1,z,2,4,6','C5,3,1,z,2,4,6','CCP5,3,1,z,2,4,6','CP5,3,1,z,2,4,6', ...
  'PCP5,3,1,z,2,4,6','P5,3,1,z,2,4,6','PPO1,2','PO1,z,2'};

opt.patch = patch_list{opt.ipatch};
center_str = center_list_str{opt.icenter};

allch = scalpChannels;
opt.patch_centers = center_list{opt.icenter};
opt.patch_centers = allch(chanind(allch, opt.patch_centers));
nCenters = length(opt.patch_centers);
[requClabs Wall neighborClabs] = getClabForLaplacian(strukt('clab',scalpChannels), ...
  'clab', opt.patch_centers, 'filter_type', opt.patch, 'require_complete_neighborhood', opt.require_complete_neighborhood);
centers_to_rm = [];
for ic = 1:nCenters
  if isempty(neighborClabs{ic})
    centers_to_rm(end+1) = ic;
  end
end
neighborClabs(centers_to_rm) = [];
opt.patch_centers(centers_to_rm) = [];
nCenters = length(opt.patch_centers);

for ic = 1:nCenters
  opt.patch_clab{ic} = cat(2, opt.patch_centers{ic}, neighborClabs{ic});
end

center_str_for_xv = ['{''' opt.patch_centers{1} ''''];
for ilap = 2:nCenters
  center_str_for_xv = cat(2,center_str_for_xv, [',''' opt.patch_centers{ilap} '''']);
end
center_str_for_xv = [center_str_for_xv '}'];

lap_opt = copy_struct(opt, 'lap_selectPolicy', 'lap_score', 'nLap', 'require_complete_neighborhood');
lap_opt.filter_type = opt.patch;
lap_opt.clab = opt.patch_centers;
cspp_opt = copy_struct(opt, 'patch_selectPolicy', 'patch_score', 'nPatPerPatch', ...
  'csp_score', 'csp_selectPolicy', 'nPat', 'covPolicy','require_complete_neighborhood','patch','patch_centers','patch_clab');
csp_opt = strukt('selectPolicy', opt.patch_selectPolicy, ...
  'score', opt.patch_score, ...
  'patterns', opt.nCSP, ...
  'covPolicy', opt.covPolicy);

%% Get list of experiments
dd= [BCI_DIR 'investigation/studies/' opt.StudyName '/'];

if isequal(opt.StudyName, 'vitalbci_season1')
  lists = {'session_list', 'session_list_tuebingen'};
else
  lists = {'session_list'};
end
subdir_list= {};
for ll= 1:length(lists),
  sdl= textread([dd lists{ll}], '%s');
  subdir_list= cat(1, subdir_list, sdl);
end

S= load([DATA_DIR 'results/' opt.StudyName '/performance']);
decent_fb= find(S.acc_f>70);
subdir_list= S.subdir_list(decent_fb);
subdir_sel= {{},{},{}};
ival= [750 3750];

res_dir = [DATA_DIR 'subject_independent_classifiers/season12/'];

suffix = [center_str '_' opt.patch '_'];

if isequal(opt.meth, 'cspp')
  suffix_cnt = [int2str(opt.nPatPerPatch) 'patPerPatch_' center_str '_' opt.patch '_'];
else
  suffix_cnt = [center_str '_' opt.patch '_'];
end

if isequal(opt.meth, 'lap')
  if isequal(opt.lap_selectPolicy, 'all')
    suffix_cnt = [suffix_cnt opt.lap_selectPolicy int2str(opt.nLap) 'lap_'];
  else
    suffix_cnt = [suffix_cnt int2str(opt.nLap) 'lap_' opt.lap_selectPolicy '_' opt.lap_score '_'];
  end
else
  if isequal(opt.patch_selectPolicy, 'all')
    suffix_cnt = [suffix_cnt opt.patch_selectPolicy int2str(opt.nPat) 'csp_'];
  else
    suffix_cnt = [suffix_cnt int2str(opt.nPat) 'csp_' opt.patch_selectPolicy '_' opt.patch_score '_'];
  end
end

if nCenters>3 || ~isequal(opt.meth,'lap')
  band = opt.band;
else
  if isequal(opt.band, [8 35])
    band = [8 15; 16 35];
  elseif isequal(opt.band, [8 32])
    band = [8 15; 16 32];
  else
    band = [opt.band(1) ceil(diff(opt.band)/2); ceil(diff(opt.band)/2)+1 opt.band(2)];
  end
end

nBands = size(band,1);

ival = [750 3750];
bandstr = strrep(sprintf('%g-%g_', band'),'.','_');
suffix = [suffix bandstr];
suffix_cnt = [suffix_cnt bandstr];

class_list= {'left','right','foot'};
nck= nchoosek(1:3, 2);
nck(3,:)= fliplr(nck(3,:));

classy= {'RLDAshrink', 'scaling', 1, 'store_means', 1, 'store_invcov', 1, 'store_extinvcov', 1};

for ci = 1:size(nck,1),

  classes = class_list(nck(ci,:));
  clstag = [upper(classes{1}(1)), upper(classes{2}(1))];
  disp(clstag)
  fileIn = ['cnt_for_common_classifier_' suffix_cnt clstag '_' opt.meth];
  load([res_dir fileIn], 'epo','mrk');
  cspp_opt.fs = epo.fs;
  mrk_cl = mrk;
  fv = epo;
  clear epo mrk
  %   mrk_cl= mrk_selectClasses(mrk, classes);
  %   fv = cntToEpo(cnt, mrk_cl, ival);

  switch opt.meth
    case 'lap'
      proc.train = ['[fv, W] = proc_selectLaps(fv, ''nLap'', ' int2str(lap_opt.nLap) ', ''lap_selectPolicy'', ''' ...
        lap_opt.lap_selectPolicy ''', ''lap_score'', ''' lap_opt.lap_score ''', ''clab'', ' center_str_for_xv ', ''filter_type'', ''' ...
        lap_opt.filter_type ''', ''require_complete_neighborhood'', ' int2str(lap_opt.require_complete_neighborhood) '); ' 'fv=proc_variance(fv); ' ...
        'fv = proc_logarithm(fv);'];
    case 'cspp'
      proc.train= ['[fv,W]= proc_cspp_auto(fv, ''nPat'', ' int2str(cspp_opt.nPat) ', ''require_complete_neighborhood'', ' ...
        int2str(cspp_opt.require_complete_neighborhood) ', ''patch_selectPolicy'', ''' ...
        cspp_opt.patch_selectPolicy ''', ''patch_score'', ''' cspp_opt.patch_score ''', ''nPatPerPatch'', ' ...
        int2str(cspp_opt.nPatPerPatch) ', ''patch'', ''' cspp_opt.patch ''', ''patch_centers'', ' center_str_for_xv '); ' ...
        'fv = proc_variance(fv); ' 'fv = proc_logarithm(fv);'];
    case 'csp'
      proc.train= ['[fv,W]= proc_csp_auto(fv, ''patterns'', ' int2str(csp_opt.patterns) ', ''csp_selectPolicy'', ''' ...
        csp_opt.selectPolicy ''', ''csp_score'', ''' csp_opt.score '''); ' 'fv = proc_variance(fv); ' ...
        'fv = proc_logarithm(fv);'];
  end

  proc.apply = ['fv = proc_linearDerivation(fv, W); ' ...
    'fv = proc_variance(fv); ' ...
    'fv = proc_logarithm(fv);'];
  proc.memo= {'W'};

  opt_xv.proc.train = proc.train;
  opt_xv.proc.apply = proc.apply;
  opt_xv.proc.memo = proc.memo;

  disp('cross validation')

  [loss_xv, loss_xv_std, outxv] = xvalidation(fv, classy, opt_xv);

  switch opt.meth
    case 'lap'
      [fv, W]= proc_selectLaps(fv, lap_opt);
    case 'cspp'
      [fv, W]= proc_cspp_auto(fv, cspp_opt);
    case 'csp'
      [fv, W]= proc_csp_auto(fv, csp_opt);
  end

  fv = proc_variance(fv);
  fv = proc_logarithm(fv);

  fprintf('<%s>: XV error %5.1f%%\n', clstag, 100*loss_xv);

  disp('train the classifier')
  C = trainClassifier(fv, classy);
  out = applyClassifier(fv, 'RLDAshrink', C);
  loss_tr = 100*mean(loss_0_1(fv.y, out));
  loss_roc_tr = loss_rocArea(fv.y, out);
  fprintf('<%s>: %5.1f%%\n', clstag, 100*loss_roc_tr);

  if nBands>1
    for ib = 1:nBands
      [filt_b{ib} filt_a{ib}]= butter(5, band(ib,:)/ct.fs*2);
    end
  else
    [filt_b filt_a]= butter(5, opt.band/ct.fs*2);
  end
  
  fileOut = [opt.meth '_' suffix clstag];

%% setup_opts
  setup_opts = cspp_opt;
  setup_opts.classes = classes;
  setup_opts.band = band;
  setup_opts.filt_b = filt_b;
  setup_opts.filt_a = filt_a;
  setup_opts.ival = ival;
  setup_opts.model = classy;
  setup_opts.clab = fv.origClab;

%% analyze
  analyze.band = band;
  analyze.ival = ival;
  analyze.clab = fv.origClab;
  analyze.spat_w = W;
  analyze.features.patch = cspp_opt.patch;
  analyze.features.patch_centers = cspp_opt.patch_centers;
  analyze.features.patch_clab = cspp_opt.patch_clab;
  analyze.features = fv;
  analyze.message = sprintf('%s training loss= %2.2f\n%s training roc loss= %2.2f', opt.meth, loss_tr,opt.meth,loss_roc_tr);

  save([res_dir fileOut], 'C', 'setup_opts','analyze');
      
  disp(['save in ' res_dir fileOut])


end

clear cnt fv
