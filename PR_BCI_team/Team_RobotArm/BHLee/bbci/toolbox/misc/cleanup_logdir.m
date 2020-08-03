function cleanup_logdir(logdir)
%CLEANUP_LOGDIR - Cleans up a feedback log directory from wrong log files
%
%Synopsis:
% clean_up_logdir(LOGDIR)
%
%Arguments:
% LOGDIR: e.g., 'VPcm_06_06_13/imag_VPcm_setup_001_log' which is taken
%    relative to global EEG_RAW_DIR, or a respective absolute path

% Author(s): Benjamin Blankertz

global EEG_RAW_DIR

if logdir(end)~='/',
  logdir= [logdir '/'];
end
sp= find(logdir=='/');
% prevent double '/' from spoiling the directory parsing
logdir(sp(find(diff(sp)==1))) = [];
sp= find(logdir=='/');

if length(sp)==1,
  error('haeh?');
end
if length(sp)>2,
  subdir= logdir(sp(end-2)+1:sp(end-1)-1);
  setupdir= logdir(sp(end-1)+1:end-1);
elseif length(sp)==2,
  subdir= logdir(1:sp(1)-1);
  setupdir= logdir(sp(1)+1:end-1);
end
if (isunix & (logdir(1)~=filesep)) | (ispc & (logdir(2)~=':')),
  logdir= [EEG_RAW_DIR logdir];
end
is= min(find(subdir=='_'));
datstr= subdir(is+[1 2 4 5 7 8]);

dd= dir([logdir '/*.log']);
for ii= 1:length(dd),
  logname= dd(ii).name;
  is= find(logname=='_');
  str= logname(is(end-6)+1:end);
  datim= sscanf(str, '%d_%d_%d_%d_%d_%d.log');
  tempdir_log= sprintf('%02d_%02d_%02d', datim(1)-2000, datim(2), datim(3));
  datstr_log= tempdir_log([1 2 4 5 7 8]);
  if ~strcmp(datstr_log, datstr),
    trgdir= [EEG_RAW_DIR 'temp/' setupdir '/' tempdir_log];
    fprintf('moving <%s> to %s/\n', logname, trgdir);
    if isunix,
      if ~exist(trgdir, 'dir'),
        mkdir([EEG_RAW_DIR 'temp/'], setupdir);
        mkdir([EEG_RAW_DIR 'temp/' setupdir], tempdir_log);
      end
      cmd= sprintf('mv %s %s/', [logdir logname], trgdir);
      [ret,why]= unix(cmd);
      if ret,
        warning(sprintf('mv failed: %s', why));
      end
      fbname= logname(1:end-8);
      fbnr= logname(end-6:end-4);
      logfbname= [fbname '_fb_opt_' fbnr '.mat'];
      cmd= sprintf('mv %s %s/', [logdir logfbname], trgdir);
      [ret,why]= unix(cmd);
      if ret,
        warning(sprintf('mv failed: %s', why));
      end
    else
      error('to be implemented: please do it');
    end
  end
end
