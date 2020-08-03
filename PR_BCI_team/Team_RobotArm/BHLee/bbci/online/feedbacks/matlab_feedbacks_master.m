function [run,feedback_opt] = matlab_feedbacks_master(feedback_opt,control_communication);
% 
% Realization of framework for multi-player feedback mode. 
% Master function sends commands of the form set... to clients who 
% do the visualization in an extra figure.
%
% matlab_feedbacks_master is called by matlab_feedbacks
%
% 11.Jan 2006 Guido, Michael

run = true;

%open the control port
if ~isfield(feedback_opt,'player1') | ~isfield(feedback_opt.player1,'fb_port')    
  feedback_opt.player1.fb_port = 12489;
end

if ~isfield(feedback_opt,'player2') | ~isfield(feedback_opt.player2,'fb_port')    
  feedback_opt.player2.fb_port = 12488;
end

timeout = 2;
master_port = get_data_udp([feedback_opt.player1.fb_port,feedback_opt.player2.fb_port]);

feedback_opt.reset = 1;
loop = true;
tic;
timli = toc;

mar = [-1,-1];

timla = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here comes the actual loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while loop
    
    pack = cell(1,2);
    % get the classifier data; including log number and block number.
    pack{1}= get_data_udp(master_port(1),0);
    pack{2}= get_data_udp(master_port(2),0);
    
    dat = cell(1,2);

    if ~isempty(pack{1}) | ~isempty(pack{2})
        c = cell(1,2);
        c{1} = pack{1};
        c{2} = pack{2};
        while ~isempty(c{1}) | ~isempty(c{2})
            for pl = 1:2
                if ~isempty(c{pl})
                    a = c{pl};% read header of packet
                    lg_no = a(1); % number of log file
                    bl_no = a(2); % block number
                    timestamp = a(3);
                    pa = a(4); % How many parallel port markers are needed?
                    if feedback_opt.log
                        % do some logging.
                        do_set(sprintf('counter%d',pl),bl_no,timestamp,lg_no);
                    end
                    for i = 1:pa % send markers 
                        do_set(a(i+4)); 
                    end
                    dat{pl} = a(pa+5:end);
                    
                    % Anything lost during the transmission?
                    if bl_no>mar(pl)+1 & mar(pl)>=0
                        fprintf('Player: %d, %i packages lost\n',pl,bl_no-mar(pl)-1);
                    end
                    mar(pl) = bl_no; %remember old block number
                    pack{pl} = c{pl};
                end
            end
            c{1} = get_data_udp(master_port(1),0);
            c{2} = get_data_udp(master_port(2),0);
        end
    end
    % dat1 and dat2 could be both empty, one could be empty, or none
    
    % Call the feedback visualization function.
%     if isempty(dat{1}) & isempty(dat{2})
%       to = toc;
%       if to-timli>0.04
%         feedback_opt = feval(feedback_opt.type,feedback_opt,dat{1},dat{2});
%         timli  = to;
%       end
%   else
%     timli = toc;  
    feedback_opt = feval(feedback_opt.type,feedback_opt,dat{1},dat{2});
%   end
    if isfield(feedback_opt,'graphic_master')
      feedback_opt = rmfield(feedback_opt,'graphic_master');    
    end

    if isempty(dat{1}) & isempty(dat{2})
      if isempty(timla)
        timla = toc;
      elseif toc-timla>timeout
        loop = false;
        feedback_opt = rmfield(feedback_opt,'type');
      end
    else
      timla = [];
    end
    % Any new setup variables?
    get_setup;
    if ~isempty(new_setup)
      do_set(235);
      feedback_opt.changed = 1;
    end
    if isempty(new_setup) & isempty(dat{1}) & isempty(dat{2}) & rand>0.99
        pause(0.0001);
    end
end

  
get_data_udp(master_port);
    
if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
  do_set(210);
end
do_set('exit',run);

pause(1);
