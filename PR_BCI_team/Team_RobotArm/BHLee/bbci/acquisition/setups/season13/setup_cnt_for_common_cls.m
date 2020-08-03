function prepare_for_common_classifier = prepare_for_common_classifier(varargin)

global EEG_MAT_DIR DATA_DIR
  
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

center_str = center_list_str{opt.icenter};
opt.patch = patch_list{opt.ipatch};

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

%   if opt.require_complete_neighborhood
%     suffix = [suffix 'rcn_'];
%   end

if nCenters>3 || ~isequal(opt.meth,'lap')
  band = opt.band;
else
  if isequal(opt.band, [8 35])
    band = [8 15; 16 35];
  elseif  isequal(opt.band, [8 32])
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
fileIn = [res_dir 'subdir_list_for_common_classifier_' suffix opt.meth];

load(fileIn);

class_list= {'left','right','foot'};
nck= nchoosek(1:3, 2);
nck(3,:)= fliplr(nck(3,:));
ival = [750 3750];

for ci = 1:size(nck,1),
  
  classes = class_list(nck(ci,:));
  clstag = [upper(classes{1}(1)), upper(classes{2}(1))];
  subdir_list = subdir_sel{ci};
  cnt= [];
  epo = [];
  mrk= [];
  disp(clstag)
  
  for vp = 1:length(subdir_list),
    
    subdir = subdir_list{vp};
    disp(subdir)
    sbj = subdir(1:find(subdir=='_', 1, 'first')-1);
    file_name = [subdir '/imag_arrow' sbj];
    [ct, mk, mnt] = eegfile_loadMatlab(file_name, 'clab', requClabs);    
    ct = tmproc_fixClabInconsistency(ct);
    if nBands>1
      for ib = 1:nBands
        [filt_b{ib} filt_a{ib}]= butter(5, band(ib,:)/ct.fs*2);
      end
      origClab = ct.clab;
      ct = proc_channelwise(ct,'filterbank',filt_b, filt_a);
      ct.origClab = origClab;
    else
      [filt_b filt_a]= butter(5, opt.band/ct.fs*2);
      ct = proc_channelwise(ct,'filt',filt_b, filt_a);
    end
    mrk_cl= mrk_selectClasses(mk, classes);
    ep = cntToEpo(ct, mrk_cl, ival);
    try
      if vp == 1
        cnt = proc_selectChannels(ct, 1);
        mrk = mrk_cl;
        epo = ep;
      else
        [cnt, mrk] = proc_appendCnt(proc_selectChannels(cnt,1), proc_selectChannels(ct, 1), mrk, mrk_cl, 'channelwise',1);
        epo = proc_appendEpochs(epo, ep);
      end
    catch
      subdir_sel{ci} = subdir_sel{ci}(1:vp-1);
      loss{ci} = loss{ci}(1:vp-1);
      disp(['processed ' int2str(length(subdir_list)-vp) ' vps less because of memory problems'])
      save(fileIn, 'subdir_sel','loss');
      break;
    end
  end
    
  fileOut = ['cnt_for_common_classifier_' suffix clstag '_' opt.meth];
    
  disp([res_dir fileOut])
  save([res_dir fileOut], 'epo','mrk','mnt');
  
end

clear cnt ct
