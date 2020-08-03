function flag = interrupt_timer(fig, marker);

flag = true;

persistent tim list bvv
list

if nargin>1;
  if isempty(tim)
    tim = timer('StartDelay',1, ...
                'ExecutionMode','fixedSpacing', ...
                'Period',1, ...
                'TimerFcn',sprintf('interrupt_timer;'), ...
                'BusyMode','queue');
    start(tim);
    list = [];
    [dum,host] = system('hostname');
    host(end) = [];
%    try
      acquire_bv('close');
      bvv = acquire_bv(100,host); 
%    end
  end
  if ~isempty(list) & ismember(fig,list(:,1))
    flag = false;
  else
    list= [list;[fig,marker]];
  end
  fprintf('Timer started\n');
  return;
end

if nargin==1,
%  if isequal(get(tim, 'Running'), 'off'),
%    fprintf('Timer died already.\n');
%  end
  if isempty(list),
    % this should not happen
    fprintf('List was deleted??!\n');
  else
    ind = find(list(:,1)==fig);
    list(ind,:) = [];
  end
  if isempty(list)
%    stop(tim);
    delete(tim);
    fprintf('Timer stopped\n');
  end
  pause(0.5);
  acquire_bv('close');
  return;
end

markerToken = [];

try 
  [currData, block, markerPos, markerToken, markerDescr] = ...
           acquire_bv(bvv);

for i = 1:length(markerToken)
  mrk = str2num(markerToken{i}(2:end));
  if ismember(mrk,list(:,2))
    [dum,loc] = ismember(mrk,list(:,2));
    ind = list(find(list(:,2)==mrk),1);
    ax = control_gui_queue(ind,'get_general_ax');
    enable_stat = set_enable(ind,'off');
    drawnow;
    set(ax.interrupt,'String','Interrupt!!!');
    pause(1);
    activate_all_entries(ind);
    interrupt_saving(ind);
    set_enable(ind,enable_stat);
  end
end

catch
  fprintf('could not get data in timer fcn.\n');
end



  