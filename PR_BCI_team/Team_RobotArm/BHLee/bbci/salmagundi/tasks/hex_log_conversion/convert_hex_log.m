function convert_hex_log(logno,log_dir,cl_log_dir,varargin)
% extract feedback signals (cnt-type and mrk-type) from logfiles.
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt,...
		   'start',0,...
		   'stop',inf,...
		   'save',[]);
fprintf('File: hexawrite %i\n',logno);
[fb_opt,counter,init_file] = load_log('hexawrite',logno);

%prepare cnt and mrk.

% browse through logfile
count= floor(opt.start/1000*fb_opt.fs);
sample = 1;
memo_frame = 0;
out = load_log;
cnt = strukt('x',zeros(1,6),'clab',{'Blocktime','arrow_ang','arrow_len','sidebar','cls','udp'},'fs',fb_opt.fs);
mrk = strukt('pos',[],'toe',{},'fs',fb_opt.fs);
arrow_len = nan;
arrow_ang = nan;
sidebar_y = nan;
blocktime = nan;
while ~isempty(out)
  if isstruct(out)
    % blocktime
    blocktime = out.time;
    out = load_log;
    continue;
  end
  if iscell(out)
    if strcmp(out{1},'BLOCKTIME'),
      % enter into cnt
      blocktime = out{2};
      out = load_log;
      continue
    end
    %otherwise: 'counter'.
    frameno= out{2};
    msec= frameno*1000/fb_opt.fs;
    fprintf('\r%010.3f ', msec/1000);
    if msec > opt.stop*1000,
      break;
    end
    if msec <= opt.start*1000,
      %do nothing.
      sample = 1;
    else
      while frameno>count+1,
        % some frames must be drawn with the same setting.
        count= count+1;
	if sample>1
	  % don't reset sample at the beginning!
	  sample = sample+1;
	end
        if opt.save,
          % STORE FRAME
	  cnt.x(sample,1) = blocktime;
	  cnt.x(sample,2) = arrow_ang;
	  cnt.x(sample,3) = arrow_len;
	  cnt.x(sample,4) = sidebar_y;
        end
      end
      if frameno>count, 
        % Next frame needs to be drawn.
        count=count+1;
	sample = sample+1;
        if opt.save,
          %STORE FRAME
	  cnt.x(sample,2) = arrow_ang;
	  cnt.x(sample,3) = arrow_len;
	  cnt.x(sample,4) = sidebar_y;
        end
      end
    end
    % update all interesting objects: arrow_len,arrow_ang,sidebar.
    switch out{4}
     case 52
      % sidebar
      if strcmp(out{5},'YData')
	sidebar_y = out{6}(3);
      end
     case 58
      % arrow
      if strcmp(out{5},'XData')|strcmp(out{5},'YData')
	arrow_x = out{6};
	arrow_y = out{8};
	foot = mean([arrow_x([1 end]); arrow_y([1 end])],2);
	tip = [arrow_x(4);arrow_y(4)];
	arrow_len = norm(tip-foot);
	arrow_ang = acos((tip(1)-foot(1))/arrow_len);
	if tip(2)<foot(2)
	  %lower half
	  arrow_ang = -arrow_ang;
	end
      end
     case 63
      % the string buffer.
      % put a marker into mrk-struct.
      if strcmp(out{5},'String')
	mrk.pos(end+1) = sample+1;
	mrk.toe{end+1} = out{6};
      end
    end
  end
  out = load_log;
end

% fill in blocktime!
bl_ind = find(cnt.x(:,1));
for ii = 1:(length(bl_ind)-1)
  cnt.x((bl_ind(ii)+1):(bl_ind(ii+1)-1),1) = (cnt.x((bl_ind(ii)),1)+1000/cnt.fs):(1000/cnt.fs):(cnt.x((bl_ind(ii)),1)+1000/cnt.fs*(bl_ind(ii+1)-bl_ind(ii)-1));
end
cnt.x(1:(bl_ind(1)),1) = cnt.x(bl_ind(1),1);
cnt.x((bl_ind(end)):end,1) = cnt.x(bl_ind(end),1):(1000/cnt.fs):(cnt.x(bl_ind(end),1)+1000/cnt.fs*(size(cnt.x,1)-bl_ind(end)));

% Add classifier logfiles:

%find out the time:
fid = fopen([log_dir sprintf('feedback_hexawrite_%3.d.log',logno)],'r');
line = fgetl(fid);
line(1:length('Feedback started at ')) = [];
line = line(1:19);
dtnum = datenum(str2num(line(1:4)),str2num(line(6:7)), str2num(line(9:10)), str2num(line(12:13)), str2num(line(15:16)), str2num(line(18:19)));

% also parse the date of the cl_logs:
fprintf('\r Loading the classifier logs');
d = dir([cl_log_dir '*.log']);
upper_min = 0;
for ii = 1:length(d)
  if ~d(ii).isdir
    line = d(ii).name;
    if isempty(findstr(d(ii).name,'basket'))
      dtnum_new = nan;
      continue;
    end
    line(1:length('basket_')) = [];
    dtnum_new(ii) = datenum(str2num(line(1:4)),str2num(line(6:7)), str2num(line(9:10)), str2num(line(12:13)), str2num(line(15:16)), str2num(line(18:19)));
  end
end
% two candidates for logfiles: one before and the other after the beginning
% of the feedback recording.
ind = find(dtnum_new<dtnum);
[ma,ma_ind] = max(dtnum_new(ind));
if ~isempty(ma_ind)
  cl_log1 = d(ind(ma_ind)).name;
end
ind = find(dtnum_new>dtnum);
[mi,mi_ind] = min(dtnum_new(ind));
if ~isempty(mi_ind)
  cl_log2 = d(ind(mi_ind)).name;
end

% hier weiter: logfiles laden/aneinanderhaengen. 
if ~isempty(ma_ind)
  log = load_logfile([cl_log_dir cl_log1]);
  % is this the right file?
  if log.udp.pos(end)*1000/log.fs<cnt.x(1,1)
    warning('First log too early - loading other file.');
    try
      log = load_logfile([cl_log_dir cl_log2]);
    catch
      error('Problems loading second log file.');
    end
  end
else
  warning('First log not found - loading other file.');
  try
    log = load_logfile([cl_log_dir cl_log2]);
  catch
    error('Problems loading second log file.');
  end
end
if log.udp.pos(1)*1000/log.fs>cnt.x(end,1)
  error('Second log too late.');
end
cnt_pointer = 1;
%cnt.clab{5} = 'cls';
%cnt.clab{6} = 'udp';
cnt.x(1:size(cnt.x,1),5:6) = nan*ones(size(cnt.x,1),2);
if length(log.cls.pos)~=length(log.udp.pos)
  warning('log.cls and log.udp have different length');
end
for ii = 1:length(log.udp.pos)
  % copy udp-signal and cls-signal.
  if log.udp.pos(ii)*1000/log.fs==cnt.x(cnt_pointer,1)
    cnt.x(cnt_pointer,5) = log.cls.values(ii);
    cnt.x(cnt_pointer,6) = log.udp.values(ii);
  end
  while log.udp.pos(ii)*1000/log.fs>=cnt.x(cnt_pointer,1)
    cnt_pointer = cnt_pointer+1;
    if cnt_pointer>size(cnt.x,1)
      break;
    end
  end
  if cnt_pointer>size(cnt.x,1)
    break;
  end
end

% Extract Markers?
switch log.setup.opt.player
 case 1
  mrk_pp = mrk_selectEvents(log.mrk, find(ismember(log.mrk.toe,-[10:26])));
 case 2
  mrk_pp = mrk_selectEvents(log.mrk, find(ismember(log.mrk.toe,[10:26])));
end
mrk_pp = mrk_selectEvents(mrk_pp, find(mrk_pp.pos*1000/log.fs>=cnt.x(1,1)&mrk_pp.pos*1000/log.fs<=cnt.x(end,1)));
for ii = 1:length(mrk_pp.pos)
  [mi,mi_ind] = min(abs(cnt.x(:,1)-mrk_pp.pos(ii)*1000/log.fs));
  mrk_pp.pos(ii) = mi_ind;
end
mrk.fs = 25;

% save the results:
if opt.save
  save(opt.save,'cnt','mrk','mrk_pp');
end
% Results:
fprintf('\r Size(cnt.x,1): %i\n',size(cnt.x,1));
fprintf('Nan-Samples(CLS): %i\n',length(find(isnan(cnt.x(:,5)))));
fprintf('Nan-Samples(UDP): %i\n',length(find(isnan(cnt.x(:,6)))));

