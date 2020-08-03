function bbci_convertRawToMat(subdir, varargin)
%BBCI_CONVERTRAWTOMAT - Convert EEG Data from BrainVision to Matlab Format
%
%Synopsis:
% bbci_convertRawToMat
% bbci_convertRawToMat(SUBDIR)
% bbci_convertRawToMat(SUBDIR, 'Property',Value, ...)
%
%Arguments:
% SUBDIR: Name of the directory that contains the data sets that are to be
%    converted. If SUBDIR is a cell array of directory names,
%    prepare_bbci_bet will process all. If SUBDIR is a patterns as
%    recognized by STRPATTERNMATCH data sets from all matching directories
%    will be processed. If SUBDIR is empty (or nonexisting) data set
%    recorded today will be processed.
%
%Properties:
% 'filespec': Specification of which files should get processed,
%    default: '*'.
% 'fs': Sampling rate, Default: 100.
% 'processing': 'none' or 'cut50'.
%
%Returns:
% nothing.

%blanker@cs.tu-berlin.de

global EEG_RAW_DIR 


% default channel layout
def_grd= sprintf(['scale,F3,Fz,F4,legend\n' ...
                  'C3,C1,Cz,C2,C4\n' ...
                  'CP3,CP1,CPz,CP2,CP4\n' ...
                  'P5,P3,Pz,P4,P6']);

%% if no subdir is given, look for today's experiment(s)
if nargin==0 | isempty(subdir) | isequal(subdir','today'),
  dd= dir(EEG_RAW_DIR);
  subdir_list= {dd.name};
  today_str= datestr(now, 'yy/mm/dd');
  today_str(find(today_str==filesep))= '_';
  iToday= strpatternmatch(['*' today_str], subdir_list);
  subdir= subdir_list(iToday);
end


%% if subdir is a cell array, call this function for each cell entry
if iscell(subdir),
  for dd= 1:length(subdir),
    bbci_convertRawToMat(subdir{dd}, varargin{:});
  end
  return;
end

%% if subdir is a pattern (contains a '*') do auto completion
if ismember('*', subdir),
  dd= dir(EEG_RAW_DIR);
  subdir_list= {dd.name};
  iMatch= strpatternmatch(subdir, subdir_list);
  subdir= subdir_list(iMatch);
  if iscell(subdir),
    bbci_convertRawToMat(subdir, varargin{:});
    return;
  end
end

if ~strcmp(subdir(end), filesep)
	subdir = sprintf('%s%s',subdir,filesep);
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'filespec', '*', ...
                 'classDef', [], ...
                 'fs', 100, ...
                 'grd', def_grd, ...
                 'processing', 'cut50', ...
                 'appendix', 'cut50');

if iscell(opt.filespec),
  for ii= 1:length(opt.filespec),
    bbci_convertRawtoMat(subdir, opt, 'filespec',opt.filespec{ii});
  end
  return;
end

if strcmp(opt.processing, 'none') & isdefault.appendix,
  opt.appendix= '';
end
if ~isempty(opt.appendix) & opt.appendix(1)~='_',
  opt.appendix= ['_' opt.appendix];
end

dd= dir([EEG_RAW_DIR subdir opt.filespec '.eeg']);
file_list= apply_cellwise({dd.name}, inline('x(1:end-4)','x'));

idx= strpatternmatch('impedances*', file_list);
file_list(idx)= [];

%S= getClassifierLog(subdir);
% if isempty(S),
%   fprintf('No classifier log file found.\n');
% end
S= [];


for ff= 1:length(file_list),
  file= file_list{ff};
  fprintf('processing %s\n', file);
  filename= [subdir file];
  
  clear cnt
 
  [mrk_orig, fs_orig]= eegfile_readBVmarkers(filename);
  var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig};
  if ~isempty(S),
    bbci= S.bbci;
    var_list= cat(2, var_list, 'bbci',bbci);
  end
%  Mrk= readMarkerTable(filename, opt.fs);
%  mrk= makeClassMarkers(Mrk, opt.classDef,0,0);
  mrk= bvmrk2mrk(mrk_orig, fs_orig);
  if ~isempty(opt.classDef),
    mrk= mrk_defineClasses(mrk, opt.classDef);
  end
  mrk= mrk_resample(mrk, opt.fs);
  hdr= eegfile_readBVheader(filename);
  clab= hdr.clab;
  if isfield(opt, 'mnt'),
    mnt= opt.mnt;
  else
    mnt= setElectrodeMontage(clab);
    mnt= mnt_setGrid(mnt, opt.grd);
  end
  
  switch(lower(opt.processing)),
    
   case 'none',
    cnt= eegfile_loadBV(filename, 'fs',opt.fs);
    cnt.title= filename;
    eegfile_saveMatlab(filename, cnt, mrk, mnt, ...
                       'channelwise',1, ...
                       'format','int16', ...
                       'resolution', 0.1, ...
                       'vars', var_list);
    
   case 'cut50',
    emgChans= chanind(clab, 'EMG*');
    nonEMG= chanind(clab, 'not', 'EMG*');
    proc.chans= {nonEMG, emgChans};
    lag= fs_orig/opt.fs;
    if lag~=round(lag),
      error('fs needs to be an integer divisor of the original sampling rate');
    end
    
    proc.eval= {['cnt= proc_filtBackForth(cnt, ''cut50''); ' ...
                 'cnt= proc_jumpingMeans(cnt, ' int2str(lag) '); '],
                ['cnt= proc_filtBackForth(cnt, ''emg''); ' ...
                 'cnt= proc_filtNotchbyFFT(cnt); ' ...
                 'cnt= proc_rectifyChannels(cnt); ' ...
                 'cnt= proc_jumpingMeans(cnt, ' int2str(lag) '); ']};
    
    cnt= readChannelwiseProcessed(filename, proc);
    cnt.title= filename;
%    eegfile_saveMatlab([filename opt.appendix], cnt, mrk, mnt, ...
%                       'channelwise',1, ...
%                       'format','double', ...
%                       'vars', var_list);

    eegfile_saveMatlab([filename opt.appendix], cnt, mrk, mnt, ...
                       'channelwise',1, ...
                       'format','int16', ...
                       'resolution', NaN, ...
                       'vars', var_list);
   otherwise
    error('unknown option for opt.processing');
  end
  
end

if isunix,
  global EEG_MAT_DIR
  bbci_owner([EEG_MAT_DIR subdir]);
end
