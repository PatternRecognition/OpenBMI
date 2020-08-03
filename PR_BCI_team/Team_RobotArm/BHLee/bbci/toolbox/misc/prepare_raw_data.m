function prepare_raw_data(subdir, varargin)
%PREPARE_RAW_DATA - Perform Basic Preparation of EEG Data in BrainVision Format
%
%Synopsis:
% prepare_raw_data
% prepare_raw_data(SUBDIR)
% prepare_raw_data(SUBDIR, 'Property',Value, ...)
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
% 'bip_list': Cell array defining calculation of bipolar channels.
% 'filespec': Specification of which files should get processed,
%    default: '*'.
%
%Returns:
% nothing.

%blanker@cs.tu-berlin.de

global EEG_RAW_DIR DATA_DIR


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
    prepare_raw_data(subdir{dd}, varargin{:});
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
    prepare_raw_data(subdir, varargin{:});
    return;
  end
end

if ~strcmp(subdir(end), filesep)
	subdir = sprintf('%s%s',subdir,filesep);
end

TMP_DATA_DIR= '/mnt/usb/data/';
if ~exist(TMP_DATA_DIR, 'dir'),
  TMP_DATA_DIR= DATA_DIR;
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filespec', '*', ...
                  'do_not_cat', {}, ...
                  'clab', '*', ...
                  'tmpdir', [TMP_DATA_DIR 'bbciRaw_tmp/' subdir(1:end-1)], ...
                  'monodir', [TMP_DATA_DIR 'bbciRaw_mono/' subdir(1:end-1)]);

subpath= [EEG_RAW_DIR subdir];

if ~exist(opt.tmpdir, 'dir'),
  [pd,nd]= fileparts(opt.tmpdir);
  mkdir(pd,nd);
  bbci_owner(opt.tmpdir);
end
if ~exist(opt.monodir, 'dir'),
  [pd,nd]= fileparts(opt.monodir);
  mkdir(pd,nd);
  bbci_owner(opt.monodir);
end

if iscell(opt.filespec),
  dd= [];
  for ii= 1:length(opt.filespec),
    ddii= dir([subpath opt.filespec{ii} '.eeg']);
    dd= cat(1, dd, ddii);
  end
else
  dd= dir([subpath opt.filespec '.eeg']);
end
file_list= apply_cellwise({dd.name}, inline('x(1:end-4)','x'));

idx= strpatternmatch('impedances*', file_list);
file_list(idx)= [];

ff= 0;
while ff<length(file_list),
  ff= ff+1;
  file= file_list{ff};
  if ~isempty(strpatternmatch(opt.do_not_cat, file)),
    continue;
  end
  isdig= ismember(file, '0123456789');
  is= max(find(~isdig));
  file_list{ff}= [file(1:is) '*'];
  append_files= setdiff(strpatternmatch(file_list{ff}, file_list), ff);
%  regmatch= regexp(file_list, [file_list{ff}(1:end-1) '[0-9]+']);
%  append_files= find(apply_cellwise2(regmatch, inline('~isempty(x)','x')));
  iDel= strpatternmatch(opt.do_not_cat, file_list(append_files));
  append_files(iDel)= [];
  file_list(append_files)= [];
end

for ff= 1:length(file_list),
  file= file_list{ff};
  fprintf('processing %s\n', file);

  %% load signals and markers
  clear cnt eog emg
  [cnt, mrk, hdr]= eegfile_loadBV([subpath file], ...
                                  'prec',1, 'verbose',1, 'clab',opt.clab);
  finalfile= file;
  finalfile(find(finalfile=='*'))= [];
  cnt.title= [subdir finalfile];
  
  cmd= sprintf('mv %s%s.* %s', subpath, file, opt.tmpdir);
  unix_cmd(cmd, 'could not move original files to tmp dir');

  %% get the original monopolar EOG channels
  eog= proc_selectChannels(cnt, 'EOG*');
  %% and save them in a separate file
  if isempty(eog.clab),
    sprintf('no EOG channels found\n');
  else
    eegfile_writeBV(eog, mrk, ...
                    'filename',[opt.monodir '/' finalfile '_eog_mono']);
  end
  %% get the original monopolar EMG channels
  emg= proc_selectChannels(cnt, 'EMG*');
  %% and save them in a separate file
  if isempty(emg.clab),
    fprintf('no EMG channels found\n');
  else
    eegfile_writeBV(emg, mrk, ...
                    'filename',[opt.monodir '/' finalfile '_emg_mono']);
  end
  clear eog emg
  cnt= intproc_bipolarEOGEMG(cnt, opt, 'ignore_tails',0);
  eegfile_writeBV(cnt, mrk, 'filename',[subpath '/' finalfile]);

end

bbci_owner(subpath);
