function ba = do_set(cas,varargin)
% function do_set(cas,<handle,name,fb_opt>)
% or:
% function do_set(number,proplist)
% or:
% function val = do_set('get',number,str);
% 
% sets the graphical objects of the current feedback figure.
% possible calls:
% do_set('init',handle,name,fb_opt)
%                 initial call to the function. store all handles and 
%                 reset the counter.
%     handle  -   vector of handles to all graphical object whose
%                 properties should be accessible.
%     name    -   name of the feedback.
%     fb_opt  -   initial setting of the optional variables.
%                 if fb_opt is true, a logfile will be written in this 
%                 function.
% do_set(number,proplist)
%                 change the graphical properties of one object.
%     number  -   index of the handle whose property should be changed.
%     proplist-   list of property and setting, e.g. 
%                 do_set(3,'Visibility','off','XData', 3)
% do_set('pause',len)
%                 pause for a while.
% do_set('get',number,str)
%                 number see above
%                 str the value to get
%     len     -   length of the pause in s.
% do_set('+')
%                 next frame.
% do_set('exit')
%                 close the logfile and exit the feedback.
% do_set('counter',number,number3,number2)    set the counter number to number and log-number to number2
%
% do_set('trigger',number)              send number to the parallel port it will be cleared to 0 after 30 milliseconds
% or 
% do_set('trigger',number, reset)  send number to the parallel port 
%                                      if reset is 1 the parallel port will be cleared after 30 milliseconds 
%                                      if reset is 0 the parallel port will stay in this state                                      

% kraulem 04/03/04
% max sagebaum 2007/10/08 - added the trigger command 

persistent handle fb_opt count fid log_number send_str  soundies
if isempty(fb_opt)
    fb_opt = set_defaults(fb_opt,'parPort',0,'log',0);
end
if isnumeric(cas)
    if ~isempty(varargin)
        % do_set(number,proplist)
        if ~isempty(handle)
            if fb_opt.graphic_master | (isfield(fb_opt,'client_machines') & ~isempty(fb_opt.client_machines)) % client due to cebit
                send_str = cat(2,send_str,{cat(2,{cas},varargin)});
            end
            if ~fb_opt.graphic_master
                set(handle(cas),varargin{:});
            end
        end
        for i = 1:length(cas)
            if fb_opt.log
                write_log('counter',count,'number',cas(i),varargin{:});
            end
        end
    else
        try
            % Parallelportmarker.
          if fb_opt.parPort
              ppTrigger(cas);
          end
          if fb_opt.log
              write_log(cas,count);
          end
        end
    end
else
    if strcmp(cas,'sound')
%      sound = varargin{1};
%      sound_fs = varargin{2};
      name = varargin{3};
      wavplay(varargin{1},varargin{2},'async');
%      soundies = cat(1,soundies,{sound,sound_fs});
      if fb_opt.log
          write_log('sound',name,'counter',count);
      end
    elseif strcmp(cas,'init')
        % do_set(cas,handle,name,fb_opt)
        handle = varargin{1};
        name = varargin{2};
        fb_opt = varargin{3};
        fb_opt = set_defaults(fb_opt, 'graphic_master', 0);
        send_str = {};
        soundies= {};
        count=0;
        % $$$     if isfield(fb_opt,'logNumber')
        % $$$       log_number = fb_opt.logNumber;
        % $$$       write_log('LOG',log_number);
        % $$$     end
        if fb_opt.log
            if write_log('???')
              write_log('flush');
              write_log('exit');
            end
            write_log('init',name,fb_opt);
        end
        if fb_opt.graphic_master | (isfield(fb_opt,'client_machines') & ~isempty(fb_opt.client_machines))   % client due to CEBIT
          if length(varargin)>3
            client = varargin{4};
          else
            client = [fb_opt.type,'_init'];
          end
          if ~isfield(fb_opt,'client_control_ports')| isempty(fb_opt.client_control_ports)
            fb_opt.client_control_ports=12470;
          end
          if length(fb_opt.client_control_ports)==1
            fb_opt.client_control_ports = fb_opt.client_control_ports*ones(1,length(fb_opt.client_machines));
          end

          if ~isfield(fb_opt,'client_ports') | isempty(fb_opt.client_ports)
            fb_opt.client_ports=12450;
          end
          if length(fb_opt.client_ports)==1
            fb_opt.client_ports = fb_opt.client_ports*ones(1,length(fb_opt.client_machines));
          end
          if ~isfield(fb_opt,'client_player') | isempty(fb_opt.client_player)
            fb_opt.client_player;
          end
          
          if length(fb_opt.client_player)==1
            fb_opt.client_player = fb_opt.client_player*ones(1,length(fb_opt.client_machines));
          end
          
          if ~isfield(fb_opt,'client_position') | isempty(fb_opt.client_position)
            fb_opt.client_position = [0 0 800 600];
          end
          
          if size(fb_opt.client_position,1)==1
            fb_opt.client_position = repmat(fb_opt.client_position,[length(fb_opt.client_machines),1]);
          end
          
          ncm = length(fb_opt.client_machines);
          for i = 1:ncm
            send_data_udp(fb_opt.client_machines{i},...
                          fb_opt.client_control_ports(i),...
                          double(sprintf('loop=false;run=true;feedback_opt.client=true;feedback_opt.client_player = %d;feedback_opt.position = [%d %d %d %d];feedback_opt.type = ''%s'';',...
                                         fb_opt.client_player(i),fb_opt.client_position(i,:),client)));
          end
          pause(2);
          
          send_data_udp(fb_opt.client_machines,fb_opt.client_ports);
        end
        
    elseif strcmp(cas,'+')
        % do_set('+')
        count=count+1;
        if fb_opt.log
            write_log('flush');
        end
        if fb_opt.graphic_master | (isfield(fb_opt,'client_machines') & ~isempty(fb_opt.client_machines))   % client due to CEBIT
            str = double(toString(send_str));
            send_data_udp(str);
            send_str = {};
        end
        if ~isempty(soundies)
           for i = 1:size(soundies,1)
             wavplay(soundies{i,1},soundies{i,2},'async');
           end
           soundies = {};
         end

        if ~fb_opt.graphic_master
            drawnow;
        end
    elseif strcmp(cas,'comment')
        if fb_opt.log
            write_log('comment',varargin{1});
        end
    elseif strcmp(cas,'exit')
        % do_set('exit')
        if fb_opt.log
            write_log('flush');
            write_log('exit');
        end
        if (isfield(fb_opt,'graphic_master') & fb_opt.graphic_master) | (isfield(fb_opt,'client_machines') & ~isempty(fb_opt.client_machines))   % client due to cebit
          send_data_udp(fb_opt.client_machines,fb_opt.client_control_ports,...
                        double(sprintf('loop=false;run=%d;',varargin{1})));
          
          send_data_udp;
          
        end 
    elseif strcmp(cas,'get')
        ba = get(handle(varargin{1}),varargin{2});
    elseif strcmp(cas,'pause')
        % do_set('pause',len)
        pause(varargin{1});
        %the number of frames skipped by this pause:
        count = count + round(varargin{1}*fb_opt.fs);
    elseif strcmp(cas,'counter')
        count = varargin{1};
        if length(varargin)>1 
            write_log('BLOCKTIME',varargin{2}, 'LOG', varargin{3});
        end
    elseif strcmp(cas,'trigger')
      ppValue = varargin{1};
      reset = 1;
      if(length(varargin) == 2) 
        reset = varargin{2};
      end
      
      try
            % Parallelportmarker.
          if fb_opt.parPort
            if reset
              ppTrigger(ppValue);
            else 
              ppFlipper(ppValue);
            end
          end
              
          if fb_opt.log
              write_log(ppValue,count);
          end
      end
    end
end
%end
