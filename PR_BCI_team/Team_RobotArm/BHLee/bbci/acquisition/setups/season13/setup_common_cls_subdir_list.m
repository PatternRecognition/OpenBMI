function select_subjects_for_common_classifier = select_subjects_for_common_classifier(varargin)

global EEG_MAT_DIR BCI_DIR PERS_DIR DATA_DIR

opt = propertylist2struct(varargin{:});

opt = set_defaults(opt, 'meth', 'patches', ...
  'ipatch', 1, ...
  'icenter', 2, ...
  'StudyName', 'vitalbci_season1', ...
  'nPatPerPatch', 1, ...  
  'csp_selectPolicy', 'equalperclass', ...
  'csp_score', 'medianvar', ...
  'nCSP', 3, ...
  'patterns', 3, ...  
  'patch_selectPolicy', 'equalperclass', ...
  'patch_score', 'medianvar', ...
  'nLap', 6, ...  
  'lap_selectPolicy', 'equalperclass', ...
  'lap_score', 'medianvar', ...
  'covPolicy', 'average', ...  
  'require_complete_neighborhood', 1, ...
  'band', [8 32], ...  
  'cont', 0, ...  
  'cb', 1);       

%% check on the consistency of the options
if isequal(opt.meth, 'patches') && ~isequal(opt.patch_selectPolicy, 'auto') && ~isequal(opt.patch_selectPolicy, 'all')
  if isequal(opt.patch_selectPolicy, 'equalperclass') || isequal(opt.patch_selectPolicy, 'directorscut')
%     if opt.patterns == 6
%       warning('with this policy option patterns per class will be calculated -> put opt.patterns = 3')
%       opt.patterns = 3;
%     end
  else
    if opt.patterns == 3
      warning('with this policy option patterns in total will be calculated -> put opt.patterns = 3')
      opt.patterns = 6;
    end
  end
  
  if ~isempty(findstr(opt.patch_selectPolicy, 'PerArea')) && opt.patterns == 3
    warning('with this policy option patterns/3 patterns per area will be chosen -> put opt.patterns = 6')
    opt.patterns = 6;
  end
end
if isequal(opt.meth, 'lap')
  if ~isempty(findstr(opt.lap_selectPolicy, 'PerArea')) && opt.nLap == 3
    warning('with this policy option nLap/3 laplacian per area will be chosen -> put opt.patterns = 6')
    opt.nLap = 6;
  end
end
if ~ischar(opt.covPolicy) && (~isequal(opt.csp_score, 'eigenvalues') || ~isequal(opt.patch_score, 'eigenvalues'))
  warning('using the class covariance matrix for csp and pacthes calculation allows to use just eigenvalues as score -> put opt.csp_score = eigenvalues and opt.patch_scroe = eigenvalues');
  opt.csp_score = 'eigenvalues';
  opt.patch_score = 'eigenvalues';
end

center_list_str = {'c12','3c', '5c', '3c3cp', '5c3cp', '3fc3c3cp', '3fc5c3cp', '3fc5c5cp', '5fc5c5cp', ...
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
patches_opt = copy_struct(opt, 'patch_selectPolicy', 'patch_score', 'nPatPerPatch', ...
  'csp_score', 'csp_selectPolicy', 'patterns', 'covPolicy','require_complete_neighborhood','patch','patch_centers','patch_clab');

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

res_dir = [DATA_DIR 'results/' opt.StudyName '/csp_patches/subject_independent_classifiers/'];

if isequal(opt.meth, 'patches')
  suffix = [int2str(opt.nPatPerPatch) 'patPerPatch_' center_str '_' opt.patch '_'];
else
  suffix = [center_str '_' opt.patch '_'];
end

if isequal(opt.meth, 'lap')
  if isequal(opt.lap_selectPolicy, 'all')
    suffix = [suffix opt.lap_selectPolicy int2str(opt.nLap) 'lap_'];
  else
    suffix = [suffix int2str(opt.nLap) 'lap_' opt.lap_selectPolicy '_' opt.lap_score '_'];
  end
else
  if isequal(opt.patch_selectPolicy, 'all')
    suffix = [suffix opt.patch_selectPolicy int2str(opt.patterns) 'csp_'];
  else
    suffix = [suffix int2str(opt.patterns) 'csp_' opt.patch_selectPolicy '_' opt.patch_score '_'];
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

if ~opt.cb
  suffix = [suffix 'fb_'];
end

class_list= {'left','right','foot'};
nck= nchoosek(1:3, 2);
nck(3,:)= fliplr(nck(3,:));

classy= {'RLDAshrink', 'scaling', 1, 'store_means', 1, 'store_invcov', 1, 'store_extinvcov', 1};
opt_xv= strukt('sample_fcn',{'chronKfold',20}, ...
  'loss', 'rocArea', ...
  'std_of_means',0);

clear loss
nVP = length(subdir_list);

fileOut = [res_dir 'subdir_list_for_common_classifier_' suffix opt.meth];

disp(fileOut)

for vp = 1:nVP,
  
  subdir= subdir_list{vp};
  sbj= subdir(1:find(subdir=='_',1,'first')-1);    
  
  if opt.cb
    file_name= [subdir '/imag_arrow' sbj];
  else
    file_name= [subdir '/imag_fbarrow' sbj];
  end
  
  hdr= eegfile_loadMatlab(file_name, 'vars','hdr');
  ihighimp= find(max(hdr.impedances,[],1)>=50);
  highimp_clab= hdr.clab(ihighimp);
  if ~isempty(intersect(requClabs, highimp_clab)),
    fprintf('%s: required channel has bad impedance\n', sbj);
    continue;
  end
  [cnt, mrk, mnt]= eegfile_loadMatlab(file_name, 'clab', requClabs);
  cnt = tmproc_fixClabInconsistency(cnt);
  [mrk, rClab] = reject_varEventsAndChannels(cnt, mrk, [500 4500]);
  
  if ~isempty(intersect(requClabs, rClab)),
    fprintf('%s: required channel has bad impedance\n', sbj);
    continue;
  end
  
  cnt = proc_selectChannels(cnt, 'not', rClab);  
  if nBands>1
    for ib = 1:nBands
      [filt_b{ib} filt_a{ib}]= butter(5, band(ib,:)/cnt.fs*2);
    end
    cnt_flt = proc_filterbank(cnt, filt_b, filt_a);
    cnt_flt.origClab = cnt.clab;
  else
    [filt_b filt_a]= butter(5, band/cnt.fs*2);
    cnt_flt = proc_filt(cnt, filt_b, filt_a);
  end

  cnt = cnt_flt;
  clear cnt_flt
  
  patches_opt.fs = cnt.fs;
    
  for ci= 1:size(nck,1),
    
    classes= class_list(nck(ci,:));
    clstag= [upper(classes{1}(1)), upper(classes{2}(1))];
    
    mrk_cl= mrk_selectClasses(mrk, classes); 
    
    if length(mrk_cl.className) == 2
            
      fv = cntToEpo(cnt, mrk_cl, ival);
      
      switch opt.meth
        case 'lap'
          proc.train = ['[fv, W] = proc_selectLaps(fv, ''nLap'', ' int2str(lap_opt.nLap) ', ''lap_selectPolicy'', ''' ...
            lap_opt.lap_selectPolicy ''', ''lap_score'', ''' lap_opt.lap_score ''', ''clab'', ' center_str_for_xv ', ''filter_type'', ''' ...
            lap_opt.filter_type ''', ''require_complete_neighborhood'', ' int2str(lap_opt.require_complete_neighborhood) '); ' 'fv=proc_variance(fv); ' ...
            'fv = proc_logarithm(fv);'];
        case 'patches'
          proc.train= ['[fv,W]= proc_cspp_auto(fv, ''patterns'', ' int2str(patches_opt.patterns) ', ''require_complete_neighborhood'', ' ...
            int2str(patches_opt.require_complete_neighborhood) ', ''patch_selectPolicy'', ''' ...
            patches_opt.patch_selectPolicy ''', ''patch_score'', ''' patches_opt.patch_score ''', ''nPatPerPatch'', ' ...
            int2str(patches_opt.nPatPerPatch) ', ''patch'', ''' patches_opt.patch ''', ''patch_centers'', ' center_str_for_xv '); ' ...
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
      
      [losstmp] = xvalidation(fv, classy, opt_xv);
      fprintf('%5s <%s>: %5.1f%%\n', sbj, clstag, 100*losstmp);
      if losstmp < 0.3,
        subdir_sel{ci}= cat(2, subdir_sel{ci}, subdir_list(vp));
        loss{ci}(length(subdir_sel{ci})) = losstmp;
      end
    end
  end
end

disp(fileOut)
save(fileOut, 'subdir_sel','loss');
