function cnt= proc_addChannelFromLog(cnt, mrk, trj_arg, fbname)
%cnt= proc_addChannelFromLog(cnt, mrk, <trj_arg, fbname>)
%cnt= proc_addChannelFromLog(cnt, mrk, <fbname>)
%
% GLOBZ EEG_RAW_DIR LOG_DIR

if ~isfield(mrk, 'info'),
  error('info field missing in mrk structure');
end
if mrk.fs~=cnt.fs,
  error('inconsistent sampling rates');
end

if exist('trj_arg','var') & ischar(trj_arg),  %% fbname is thrid argument
  fbname= trj_arg;
  trj_arg= NaN;
end

[sub_dir, file]= fileparts(cnt.title);
if ~exist('fbname','var'),
  is= min(find(file=='_'))+1;
  ie= max(find(ismember(file,['A':'Z'])))-1;
  fbname= file(is:ie);
end

if ~exist('trj_arg','var') | (isnumeric(trj_arg) & isnan(trj_arg)),
  if ~isempty(strmatch('basket', fbname)),
    trj_arg= {{3,'xData','yData'}, 1:8, 10:30};
    fbname= 'basket';
  elseif ~isempty(strmatch('1d', fbname)),
    trj_arg= {{1,'xData','yData'}, 1:8, 10:30};
    fbname= '2d';
  elseif ~isempty(strmatch('mental_states_moving', fbname)),
    trj_arg= {{[6,7],{'xData',1},{'yData',1}}, 251, 254};
  else
    error(sprintf('argument #3 (chandef) must be specified for %s',fbname));
  end
end
chandef= trj_arg{1};

global EEG_RAW_DIR LOG_DIR
LOG_DIR= [EEG_RAW_DIR sub_dir '/log/'];

[T, nChans]= size(cnt.x);
nNewChans= length(chandef)-1;
cnt.x(:,nChans+1:nChans+nNewChans)= NaN;
for cc= 1:nNewChans,
  if iscell(chandef{cc+1})
    cnt.clab{nChans+cc}= chandef{cc+1}{1};
  else
    cnt.clab{nChans+cc}= chandef{cc+1};
  end
end

for fb= 1:size(mrk.info,2),
  logno= mrk.info(1, fb);
  fprintf('adding channel from <%s_%d>\n', fbname, logno);
  [trj, fb_opt]= trajectories(fbname, logno, trj_arg{:});
  
  lag= mrk.fs/trj.fs;
  for tt= 1:length(trj.trajectories),
    spos= mrk.info(2, fb) + trj.start(tt)*lag;
    tlen= length(trj.trajectories{tt}{1});
    iv= spos:spos+lag*tlen-1;
    tiv= repmat(1:tlen, [lag 1]);
    for cc= 1:length(trj.trajectories{tt}),
      cnt.x(iv,nChans+cc)= trj.trajectories{tt}{cc}(tiv(:));
    end
  end
end
