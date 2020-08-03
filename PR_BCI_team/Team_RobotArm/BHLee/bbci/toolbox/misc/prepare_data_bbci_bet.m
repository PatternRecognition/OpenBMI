function opt= prepare_data_bbci_bet(sub_dir, varargin)
%prepare_bbci_bet - Convert data sets from generic data format to matlab
%
%Synopsis:
% prepare_bbci_bet(SUB_DIR)
% prepare_bbci_bet(SUB_DIR, 'Property',Value, ...)
%
%Arguments:
% SUB_DIR: Name of the directory that contains the data sets that are to be
%    converted. If SUB_DIR is a cell array of directory names,
%    prepare_bbci_bet will process all. If SUB_DIR is a patterns as
%    recognized by STRPATTERNMATCH data sets from all matching directories
%    will be processed. If SUB_DIR is empty (or nonexisting) data set
%    recorded today will be processed.
%
%Properties:
% 'file' File name, file name pattern or cell arraw of such, defining
%    which files should be processed. Default: '*'.
% 'setup': Setup file number (integer) or setup file (char).
%    Default: Search EEG_RAW_DIR for non-empty subdirectory and take it.
% 'classes': Default: extract from setup file.
% 'classes_fb': Default: extract from setup file.
% 'player': player number 1 or 2. Default: extract from setup file.
% 'marker_type': 'stimulus' or 'response'. Default: 1-subject 
%    experiment:'stimulus'; 2 subject experiment: player 1 'response',
%    player 2 'stimulus'. until 060327 
%    player 1,'stimulus',player,'response' afterwards
% 'include_logfiles': If set true, log files (feedback and classifier)
%    are loaded and the information (if appropriate) is included in the
%    mrk variable.
%
%Returns:
% nothing.
%
%
%Warning: The processing of a whole directory by this functions works
% only in simple cases. Restrictions are
% - only one setup file was used.
% - ...

% Author(s): Guido Dornhege, Benjamin Blankertz

global EEG_RAW_DIR

%?? How did that put here? filesep is a matlab function.
%if isunix
%	filesep = '/';
%else
%	filesep = '\';
%end

%% this should be superficial soon ...
if ~exist('load_feedback', 'file'),
  warning('bbci_bet in not in the path. setting up bbci_bet_unstable...');
  setup_bbci_bet_unstable;
end

%% if no sub_dir is given, look for today's experiment(s)
if nargin==0 | isempty(sub_dir) | isequal(sub_dir','today'),
  dd= dir(EEG_RAW_DIR);
  sub_dir_list= {dd.name};
  today_str= datestr(now, 'yy/mm/dd');
  today_str(find(today_str==filesep))= '_';
  iToday= strpatternmatch(['*' today_str], sub_dir_list);
  sub_dir= sub_dir_list(iToday);
end


%% if sub_dir is a cell array, call this function for each cell entry
if iscell(sub_dir),
  for dd= 1:length(sub_dir),
    prepare_data_bbci_bet(sub_dir{dd}, varargin{:});
  end
  return;
end

%% if sub_dir is a pattern (contains a '*') do auto completion
if ismember('*', sub_dir),
  dd= dir(EEG_RAW_DIR);
  sub_dir_list= {dd.name};
  iMatch= strpatternmatch(sub_dir, sub_dir_list);
  sub_dir= sub_dir_list(iMatch);
  if iscell(sub_dir),
    prepare_data_bbci_bet(sub_dir, varargin{:});
    return;
  end
end


if ~strcmp(sub_dir(end), filesep)
	sub_dir = sprintf('%s%s',sub_dir,filesep);
end


is= min(find(sub_dir=='_'));
subject= sub_dir(1:is-1);
date_str= sub_dir(is+1:end-1);


opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'file', '*', ...
                 'classes', [], ...
                 'classes_fb', [], ...
                 'setup', [], ...
                 'player', 1, ...
                 'marker_type', 'stimulus', ...
                 'fs', 100, ...
                 'log_info', [], ...
                 'log_fb', [], ...
                 'include_logfiles', 0, ...
                 'processing', 'cut50');

fprintf('\n* Processing data of <%s> recorded on %s.\n\n', subject, date_str);

%% TODO: make automatic choosing of setup file more inteligent.
%% use latest creation time (?).
if isdefault.setup,
  dd= dir([EEG_RAW_DIR sub_dir '/*_log']);
  nLog= ones(1, length(dd));
  for ii= 1:length(dd),
    ds= dir([EEG_RAW_DIR sub_dir '/' dd(ii).name]);
    nLog(ii)= length(ds)-2;  %% two entries are always '.' and '..'
  end
  iLog= nLog>0;
  if sum(iLog)==1,
    opt.setup= [EEG_RAW_DIR sub_dir dd(find(iLog)).name(1:end-4)];
  elseif sum(iLog)>1,
    [mm, iSetup]= max(nLog);
%    opt.setup= dd(iSetup).name(end-6:end-4);
    opt.setup= [EEG_RAW_DIR sub_dir  dd(iSetup).name(1:end-4)];
    msg= sprintf(['More than one log subdirectory is non-empty. ' ...
                  'Choosing %s as setup file.'], dd(iSetup).name(1:end-4));
    warning(msg);
  end
end

bbci= [];
if ~isempty(opt.setup),
  if isnumeric(opt.setup),
    opt.setup= sprintf('%03d', opt.setup);
  end
  if length(opt.setup)==3,
    ddsetup= dir([EEG_RAW_DIR sub_dir '/*_setup_' opt.setup '.mat']);
    opt.setup= [EEG_RAW_DIR sub_dir ddsetup.name(1:end-4)];
  end
  if (isunix & opt.setup(1)~='/') | (~isunix & opt.setup(2)~=':'),
    opt.setup = [EEG_RAW_DIR sub_dir opt.setup];
  end
  log_files = [opt.setup '_log'];
  feedback_log_files = [subject '_' date_str filesep 'log' filesep];

  if ~exist([opt.setup '.mat'], 'file'),
    msg= sprintf(['No setup file found: <%s>\n' ...
                  'bbci information cannot be included.'], opt.setup);
    warning(msg);
  else
    if isdefault.setup,
      msg= sprintf('No setup file specified: using %s.', opt.setup);
      warning(msg);
    end
    SS = load(opt.setup);
    bbci = SS.bbci;
  end
end

if ~isempty(bbci),
  if isdefault.player,
    opt.player= bbci.player;
  else
    if opt.player~=bbci.player,
      msg= sprintf(['Given player number (%d) does not match ' ...
                    'number in setup file (%d).'], opt.player, bbci.player);
      warning(msg);
    end
  end
  if isdefault.classes,
    opt.classes= bbci.classDef(2,:);
  else
    if ~isfield(bbci,'classDef')
      msg = 'No field ''classDef'' found in bbci.';
      warning(msg);
    elseif ~isequal(opt.classes, bbci.classDef(2,:)),
      msg= sprintf(['Given classes (%s) do not match classes in ' ...
                    'setup file (%s).'], vec2str(opt.classes,'%s', filesep), ...
                   vec2str(bbci.classDef(2,:),'%s',filesep));
      warning(msg);
    end
  end
  if isdefault.classes_fb,
    opt.classes_fb= bbci.classes;
  else
    if ~isfield(bbci,'classDef')
      msg = 'No field ''classDef'' found in bbci.';
      warning(msg);
    elseif ~isequal(opt.classes_fb, bbci.classes),
      msg= sprintf(['Given classes (%s) do not match classes in ' ...
                    'setup file (%s).'], vec2str(opt.classes_fb,'%s',filesep), ...
                   vec2str(bbci.classes,'%s',filesep));
      warning(msg);
    end
  end
end

if isdefault.marker_type,
  datstr = str2num(date_str([1:2,4:5,7:8]));
  if datstr>=60327
    if opt.player==1
      opt.marker_type = 'stimulus';
    else
      opt.marker_type = 'response';
    end
  else
    if opt.player==2,
      opt.marker_type= 'stimulus';
    else
      %% check whether it was a one or two player recording    
      dd= dir(EEG_RAW_DIR);
      sub_dir_list= {dd.name};
      iMatch= strpatternmatch(['*' date_str '*'], sub_dir_list);
      if length(iMatch)==2,
        opt.marker_type= 'response';
      else
        opt.marker_type= 'stimulus';
      end
    end
  end
end

if opt.include_logfiles & ~isempty(opt.setup),
  cleanup_logdir(log_files);
  if isempty(opt.log_info) & ~isempty(opt.setup),
    log_info = load_logfile(log_files,opt.fs);
    log_info = map_logmarker(log_info, opt.marker_type);
    opt.log_info= log_info;
  else
    log_info= opt.log_info;
  end

  cleanup_fblogdir(feedback_log_files);
  if isempty(opt.log_fb) & ~isempty(opt.setup),
    log_fb = load_feedback(feedback_log_files,opt.fs);
    for i = 1:length(log_fb)
      ind = find(log_fb(i).update.counter==0);
      if ~isempty(ind)& ind(end)<length(log_fb(i).update.counter),log_fb(i).update.counter(ind) = log_fb(i).update.counter(ind(end)+1)-1;end
      ind = find(isnan(log_fb(i).update.pos));
      if ~isempty(ind)& ind(end)<length(log_fb(i).update.pos),log_fb(i).update.pos(ind) = log_fb(i).update.pos(ind(end)+1)-4;end
      ind = find(isnan(log_fb(i).update.lognumber));
      if ~isempty(ind)& ind(end)<length(log_fb(i).update.lognumber),log_fb(i).update.lognumber(ind) = log_fb(i).update.lognumber(ind(end)+1);end
    end
    opt.log_fb= log_fb;
  else
    log_fb= opt.log_fb;
  end
  if isempty(log_fb)
    warning([feedback_log_files ' does not contain feedback logfile informations!']);
  end
end

if iscell(opt.file),
  dd= [];
  for ff= 1:length(opt.file),
    dd= cat(1, dd, dir([EEG_RAW_DIR sub_dir opt.file{ff} '.eeg']));
    if isempty(dd),
      warning(sprintf('No files matching <%s> found.', opt.file{ff})); 
    end
  end
else
  dd= dir([EEG_RAW_DIR sub_dir  opt.file '.eeg']);
  if isempty(dd),
    warning(sprintf('No files matching <%s> found.', opt.file));
  end
end
file_list= {dd.name};
nDel= length([subject '.eeg']);

for fi= 1:length(file_list),
clear logf

file_type= file_list{fi}(1:end-4);
file= [sub_dir file_type];
fprintf(' - processing: %s\n', file_type);
[mrk_orig, fs_orig]= eegfile_readBVmarkers(file);
Mrk= readMarkerTable(file, opt.fs);

if strpatterncmp('impedances*', file_type),
  
  datstr = str2num(date_str([1:2,4:5,7:8]));
  if datstr<60327
    split_impedances(file, opt.player);
  end
  continue;
   
elseif strpatterncmp('imag_lett*',file_type) | ...
      strpatterncmp('imag_move*',file_type) | ...
      strpatterncmp('imag_arrow*',file_type) | ...
      strpatterncmp('imag_audi*',file_type) | ...
      strpatterncmp('imag_cebit*',file_type) | ...
      (strpatterncmp('real*',file_type) & ~strpatterncmp('real_*',file_type)),

% ----- extract marker information from off-line calibration measurement -----

  numcl = num2cell(1:length(opt.classes));
  classDef= {numcl{:}; opt.classes{:}};
  mrk= makeClassMarkers(Mrk, classDef,0,0);
  
%% ----- end

else
  if isempty(Mrk.pos),
    mrk= Mrk;
    msg= sprintf('No markers found in %s.\n', file);
  else
    mrk= mrkdefByClassDef(Mrk, file_type);
    if isnumeric(mrk) & isnan(mrk),
      ft = file_type;
      while length(ft)>0 & ~exist(['mrkdef_' ft]),
        ft = ft(1:end-1);
      end
      if length(ft)>0,
        fprintf('Using ''mrkdef_%s'' for marker definition.\n', ft); 
%        if strcmp('imag_1d', ft)
%        continue
%        end
        mrk= feval(['mrkdef_' ft], Mrk, file, opt);
        if isfield(mrk,'logf')
          logf = mrk.logf;
          mrk = rmfield(mrk,'logf');
        end
        if isfield(mrk,'flogf')
          flogf = mrk.flogf;
          mrk = rmfield(mrk,'flogf');
        end
        
      else
        msg= sprintf('Format of <%s> not known. Skipping.', file_type);
        warning(msg);
        continue;
      end
    end
  end
end


cnt= eegfile_loadBV(file, 'fs',opt.fs);

if length(cnt.clab)==1 & strcmp(lower(cnt.clab{1}),'nan') % if recording was forgotten, cnt is nan
  mnt = [];
else
  %% if preprocess_data (run on brainamp) has memory problems,
  %% some nuissance channels labeled NaC (not a channel) may be present
  cnt= proc_selectChannels(cnt, 'not', 'NaC');
  
  mnt= getElectrodePositions(cnt.clab);
  grd= sprintf('EOGh,F3,legend,F4,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nEMGl,O1,EMGf,O2,EMGr');
  mnt= setDisplayMontage(mnt, grd);
  mnt= mnt_excenterNonEEGchans(mnt, 'E*');
end

var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig};
if exist('logf','var'),
  var_list= cat(2, var_list, {'log',logf, 'feedback',flogf});
end
if ~isempty(bbci),
  var_list= cat(2, var_list, {'bbci',bbci});
end

switch(lower(opt.processing)),
    
 case 'none',
  eegfile_saveMatlab(file, cnt, mrk, mnt, ...
                     'channelwise',1, ...
                     'format','int16', ...
                     'resolution', 0.1, ...
                     'vars', var_list);
  
 case 'cut50';
  if length(cnt.clab)==1 & strcmpi(cnt.clab{1},'nan'),
    warning(sprintf('Trouble with file <%> in function %s', ...
                    file, mfilename));
    return;
  end
  clab= eegfile_readBVheader(file);
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
  
  cnt= readChannelwiseProcessed(file, proc);
  [dmy, title2]= fileparts(file);  %% extract last subdir and filename
  [dmy, title1]= fileparts(dmy);
  cnt.title= [title1 '/' title2];
  eegfile_saveMatlab([file '_cut50'], cnt, mrk, mnt, ...
                     'channelwise',1, ...
                     'format','int16', ...
                     'resolution', NaN, ...
                     'vars', var_list);
end

end

if isunix,
  global EEG_MAT_DIR
  bbci_owner([EEG_MAT_DIR sub_dir]);
end

if nargout==0,
  clear opt;
end
