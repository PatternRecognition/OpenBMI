function logs = select_logfile(log,window,ref);

if nargin<3 | isempty(ref)
  ref = 0;
end

beg = find(window(1)>log.segments.ival(:,1));
if isempty(beg)
  beg = 1;
else
  beg = beg(1);
end
en = find(window(2)<log.segments.ival(:,2));
en = en(1);



logs = struct('fs',log.fs);
logs.filename = log.filename(beg:en);

tt = log.segments.ival(beg,1);
tt2 = datenum(log.segments.time(beg,:))*24*60*60*log.fs;
tt = tt-window(1);
tt2 = tt2-tt;
tt2 = tt2/log.fs/24/60/60;

logs.time = datevec(tt2);

logs.feedback = log.feedback;

if iscell(log.setup_file)
  logs.setup_file = log.setup_file(beg:en);
else
  logs.setup_file = log.setup_file;
end
if isempty(log.setup_file)
  warning('empty setup file!');
  keyboard
end

  
logs.continuous_file_number = log.continuous_file_number(beg:en);

logs.log_variables = log.log_variables(beg:en,:);

ind = find(log.mrk.pos>=window(1) & log.mrk.pos<=window(2));
logs.mrk = struct('pos',log.mrk.pos(ind)-ref,'toe',log.mrk.toe(ind));

ind = find(log.comment.pos>=window(1) & log.comment.pos<=window(2));
logs.comment = struct('pos',log.comment.pos(ind)-ref,'comment',{log.comment.comment(ind)});

ind = find(log.msg.pos>=window(1) & log.msg.pos<=window(2));
logs.msg = struct('pos',log.msg.pos(ind)-ref,'message',{log.msg.message(ind)});

ind = find(log.cls.pos>=window(1) & log.cls.pos<=window(2));
logs.cls = struct('pos',log.cls.pos(ind)-ref,'values',{log.cls.values(:,ind)});

ind = find(log.udp.pos>=window(1) & log.udp.pos<=window(2));
logs.udp = struct('pos',log.udp.pos(ind)-ref,'values',log.udp.values(:,ind));

ind = find(log.changes.pos>=window(1) & log.changes.pos<=window(2));
logs.changes = struct('pos',log.changes.pos(ind)-ref,'time',log.changes.time(ind,:),'code',{log.changes.code(ind)});

ind = find(log.segments.ival(:,1)>=window(1) & log.segments.ival(:,2)<=window(2));
iv = log.segments.ival(ind,:);
iv(1,1) = window(1)+1;
iv(end,2) = window(2);
logs.segments = struct('ival',iv-ref,'time',cat(1,logs.time,log.segments.time(ind,:),datevec((datenum(logs.time)*24*60*60*log.fs+diff(window))/log.fs/60/60/24)));


ind = find(log.adaptation.pos>=window(1) & log.adaptation.pos<=window(2));
if isempty(ind)
  fprintf('Correcting adaptation positions.')
  % correct the adaptation positions.
  pos = [];
  for ii = 1:size(log.adaptation.time,1)
    tt_adap = datenum(log.adaptation.time(ii,:))*24*60*60*log.fs;
    tt_adap = floor(tt_adap-datenum(log.segments.time(beg,:))*24*60*60*log.fs)+window(1);
    pos = [pos tt_adap];
  end
  log.adaptation.pos = pos;
  ind = find(log.adaptation.pos>=window(1) & log.adaptation.pos<=window(2));
  disp(['New positions inside the window: ' num2str(length(ind))]);
end
logs.adaptation = struct('pos',log.adaptation.pos(ind)-ref,'time',log.adaptation.time(ind,:),'code',{log.adaptation.code(ind)});


