function cleanup_fblogdir(logdir)
%CLEANUP_FBLOGDIR - Cleans up a feedback log directory from wrong log files
%
%Synopsis:
% cleanup_fblogdir(LOGDIR)
%
%Arguments:
% LOGDIR: e.g., 'VPcm_06_06_06' or 'VPcm_06_06_06/log' which are taken
%    relative to global EEG_RAW_DIR, or a respective absolute path

% Author(s): Benjamin Blankertz

global EEG_RAW_DIR

if logdir(end)~='/',
  logdir= [logdir '/'];
end
sp= find(logdir=='/');
if length(sp)==1,
  logdir= [logdir 'log/'];
end
if length(sp)>2,
  subdir= logdir(sp(end-2)+1:sp(end-1)-1);
elseif length(sp)==2,
  subdir= logdir(1:sp(1)-1);
end
if (isunix & (logdir(1)~=filesep)) | (ispc & (logdir(2)~=':')),
  logdir= [EEG_RAW_DIR logdir];
end
is= min(find(subdir=='_'));
datstr= subdir(is+[1 2 4 5 7 8]);

dd= dir([logdir '/*.log']);
for ii= 1:length(dd),
  logname= dd(ii).name;
  fid= fopen([logdir logname], 'r');
  str= fgets(fid);
  if isequal(str, -1),   %% log file empty
    continue; 
  end
  datim= sscanf(str, 'Feedback started at %d_%d_%d_%d_%d_%d with values:');
  tempdir_log= sprintf('%02d_%02d_%02d', datim(1)-2000, datim(2), datim(3));
  datstr_log= tempdir_log([1 2 4 5 7 8]);
  if ~strcmp(datstr_log, datstr),
    trgdir= [EEG_RAW_DIR 'temp/' tempdir_log];
    fprintf('moving <%s> to %s/\n', logname, trgdir);
    if isunix,
      if ~exist(trgdir, 'dir'),
        mkdir([EEG_RAW_DIR 'temp/'], tempdir_log);
      end
      cmd= sprintf('mv %s %s/', [logdir logname], trgdir);
      [ret,why]= unix(cmd);
      if ret,
        error(sprintf('mv failed: %s', why));
      end
      fbname= logname(1:end-8);
      fbnr= logname(end-6:end-4);
      logfbname= [fbname '_fb_opt_' fbnr '.mat'];
      cmd= sprintf('mv %s %s/', [logdir logfbname], trgdir);
      [ret,why]= unix(cmd);
      if ret,
        error(sprintf('mv failed: %s', why));
      end
    else
      error('to be implemented: please do it');
    end
  end
end
