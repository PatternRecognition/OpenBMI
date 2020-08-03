function flag = parport_check(typ,machine);
global general_port_fields
if nargin<2 | isempty(machine)  if ~isempty(general_port_fields)      machine = general_port_fields(1).bvmachine;  else
      machine = 'brainamp';  end
end

if nargin<1 | isempty(typ)
  typ = 'Stimulus';
end

state = acquire_bv(100,machine);
if isempty(state),
  warning('Recorder is not in monitoring mode');
  flag = true;
  return;
end


ppTrigger(255);

tic
flag = false;
while toc<1 & ~flag
  [dum1,dum2,dum3,d] = acquire_bv(state);
  for i = 1:length(d)
    dd = d{i};
    if typ(1)==dd(1)
      dd = str2num(dd(2:end));
      if dd==255
        flag = true;
        break;
      end
    end
  end
end

acquire_bv('close');
