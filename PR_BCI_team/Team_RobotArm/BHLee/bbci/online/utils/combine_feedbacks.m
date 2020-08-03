function log = combine_feedbacks(log1,log2);

log = struct('start',cat(1,log1.start,log2.start));
if ~iscell(log1.file)
  log1.file = {log1.file};
end
if ~iscell(log2.file)
  log2.file = {log2.file};
end

log.file = unique(cat(2,log1.file,log2.file));
log.fs = log1.fs;

log.mrk = struct('pos',cat(2,log1.mrk.pos,log2.mrk.pos));
log.mrk.toe = cat(2,log1.mrk.toe,log2.mrk.toe);
log.mrk.counter = cat(2,log1.mrk.counter,log2.mrk.counter);
log.mrk.lognumber = cat(2,log1.mrk.lognumber,log2.mrk.lognumber);


log.update = struct('pos',cat(2,log1.update.pos,log2.update.pos));
log.update.object = cat(2,log1.update.object,log2.update.object);
log.update.counter = cat(2,log1.update.counter,log2.update.counter);
log.update.lognumber = cat(2,log1.update.lognumber,log2.update.lognumber);
log.update.prop = cat(2,log1.update.prop,log2.update.prop);
log.update.prop_value = cat(2,log1.update.prop_value,log2.update.prop_value);

log.initial = cat(2,log1.initial,log2.initial);

if isfield(log1,'init_file')
  log.init_file = log1.init_file;
end
