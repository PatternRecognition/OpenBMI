function [dat,state] = online_processing(cnt,state,proc,param);

dat = copyStruct(cnt,'x','clab');
dat.x = [];
dat.clab = {};

if isempty(state)
  state = cell(1,length(proc));
  for i = 1:length(proc)
    [da,state{i}] = feval(['online_' proc{i}],cnt,[],param{i}{:});
    dat.x = cat(2,dat.x,da.x);
    dat.clab = {dat.clab{:},da.clab{:}};
  end
else
  for i = 1:length(proc)
    [da,state{i}] = feval(['online_' proc{i}],cnt,state{i},param{i}{:});
    dat.x = cat(2,dat.x,da.x);
    dat.clab = {dat.clab{:},da.clab{:}};
  end
end
