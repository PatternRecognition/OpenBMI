function log = combine_logs(log1,log2);

try
  if ~iscell(log1.filename)
    log1.filename = {log1.filename};
  end
  if ~iscell(log2.filename)
    log2.filename = {log2.filename};
  end
  
  if ~iscell(log1.setup_file)
    log1.setup_file = {log1.setup_file};
  end
  if ~iscell(log2.setup_file)
    log2.setup_file = {log2.setup_file};
  end
  if isfield(log1,'setup')
    if ~iscell(log1.setup)
      log1.setup = {log1.setup};
    else
% What is this?? (BB)
%     msg = sprintf('No setup fiels in log %s', log1);
%     warning(msg);
     warning(sprintf('something strange here (%s)', mfilename));
    end
  end
  if isfield(log2,'setup')
    if ~iscell(log2.setup)
      log2.setup = {log2.setup};
    else
% What is this?? (BB)
%     msg = sprintf('No setup fiels in log %s', log2);
%     warning(msg);
     warning(sprintf('something strange here (%s)', mfilename));
    end
  end
  
  if ~isfield(log1,'comment')
    log1.comment = struct('pos',[],'comment',{{}});
  end
  if ~isfield(log2,'comment')
    log2.comment = struct('pos',[],'comment',{{}});
  end
  
  log = struct('fs',log1.fs,'filename',{unique({log1.filename{:},log2.filename{:}})});
  log.time = log1.time;
  log.feedback = log1.feedback;
  aaa = {log1.setup_file{:},log2.setup_file{:}};
  [log.setup_file,I] = unique(aaa);
  if isfield(log1,'setup')
    aaa = {log1.setup{:},log2.setup{:}};
    log.setup = {aaa{I}};
  end
  log.continuous_file_number = unique([log1.continuous_file_number,log2.continuous_file_number]);
  log.log_variables = cat(1,log1.log_variables,log2.log_variables);
  log.log_variables = log.log_variables(I,:);
  log.mrk = struct('pos',[log1.mrk.pos,log2.mrk.pos],'toe',[log1.mrk.toe,log2.mrk.toe]);
  log.msg = struct('pos',[log1.msg.pos,log2.msg.pos],'message',{cat(2,log1.msg.message,log2.msg.message)});
  log.cls = struct('pos',[log1.cls.pos,log2.cls.pos],'values',{cat(2,log1.cls.values,log2.cls.values)});
  log.udp = struct('pos',[log1.udp.pos,log2.udp.pos],'values',[log1.udp.values,log2.udp.values]);
  log.changes = struct('pos',[log1.changes.pos,log2.changes.pos],'time', cat(1,log1.changes.time,log2.changes.time),'code',{cat(2,log1.changes.code,log2.changes.code)});
  log.segments = struct('ival',[log1.segments.ival;log2.segments.ival],'time', cat(1,log1.segments.time,log2.segments.time));
  log.adaptation = struct('pos',[log1.adaptation.pos,log2.adaptation.pos],'time', cat(1,log1.adaptation.time,log2.adaptation.time),'code',{cat(2,log1.adaptation.code,log2.adaptation.code)});
  log.comment = struct('pos',[log1.comment.pos,log2.comment.pos],'comment',{cat(2,log1.comment.comment,log2.comment.comment)});
  
catch
  warning('Could not combine logfiles');
  keyboard
  log = [];
end

