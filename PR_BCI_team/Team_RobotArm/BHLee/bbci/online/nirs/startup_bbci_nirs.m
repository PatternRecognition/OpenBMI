global NEURO_DIR IDABOX_DIR DATA_DIR TEX_DIR
global BCI_DIR EEG_RAW_DIR EEG_MAT_DIR EEG_CFG_DIR 
%global EEG_FIG_DIR EEG_EXPORT_DIR EEG_IMPORT_DIR
global SOUND_DIR

%global value for the function used for data acquisition set default value
%to acquire_nirs
global acquire_func
acquire_func = @acquire_nirs;

BCI_DIR= [fileparts(fileparts(fileparts(which(mfilename)))) filesep];
TEX_DIR= [BCI_DIR(1:end-1) '_tex/'];

%% add paths of the BBCI toolbox (to the end of the path)
if isunix,
  fs= filesep;
else
  fs= ['\' filesep];  %% we need an escape character for regexp
end
tb_path= genpath([BCI_DIR 'toolbox']);
path(path, tb_path);
svn_pattern= ['.*' fs '\.svn.*'];
[dmy, idx]= path_exclude(tb_path, svn_pattern);
rmpath(tb_path(idx));
clear dmy idx
appendpathifexists([BCI_DIR 'investigation' filesep 'utils']);
appendpathifexists([BCI_DIR 'acquisition' filesep 'setups'], 'recursive',0);
appendpathifexists([BCI_DIR 'acquisition' filesep 'stimulation'], 'recursive',0);
appendpathifexists([BCI_DIR 'acquisition' filesep 'tools'], 'recursive',0);

%% define some important paths as global variables
EEG_CFG_DIR= [BCI_DIR 'toolbox/data/config/'];
SOUND_DIR= [BCI_DIR 'acquisition/data/sound/'];

EEG_RAW_DIR= [DATA_DIR 'bbciRaw/'];
%EEG_MAT_DIR= [DATA_DIR 'bbciMat/'];
%EEG_EXPORT_DIR= [DATA_DIR 'eegExport/'];
%EEG_IMPORT_DIR= [DATA_DIR 'eegImport/'];
%EEG_FIG_DIR= [TEX_DIR 'pics/'];

format long g;
format compact;
%cd(BCI_DIR);
