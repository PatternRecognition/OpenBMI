function send_trigger_vest(varargin),

% use:
%
% Initialize connection
%   send_trigger_vest('init', '127.0.0.1', 12345);
%
% Send package
%   send_trigger_vest('send', direction);
%
% Set stimulus duration
%   send_trigger_vest('duration', duration); %-1 =~ no timeout
% 
% Close connection
%   send_trigger_vest('close');
%
% Martijn Schreuder, 2010

global BCI_DIR
persistent socke int_type

    switch varargin{1}
        case 'init'
          hostname= varargin{2};
          port= varargin{3};
          path(path, [BCI_DIR 'import/tcp_udp_ip']);
          socke= pnet('udpsocket', 1111);  %% what is this port number?
          if socke==-1,
            error('udp communication failed');
          end
          pnet(socke, 'udpconnect', hostname, port);   
          int_type = 'int16';
            
        case 'send'
          if isempty(socke),
              error('Connection not properly initialized. Run init first');
          end
          if isempty(varargin{2}), 
            disp('All tactors deactivated');
          else
            subMes = sprintf('%i,', varargin{2});
            disp(sprintf('Vest tactor(s) %s activated.', subMes(1:end-1)));
          end
          
          message = zeros(1,64,int_type);
          message(varargin{2}) = 1;
          pnet(socke, 'write', message);
          pnet(socke, 'writepacket');
        
        case 'duration'
          if isempty(socke),
              error('Connection not properly initialized. Run init first');
          end
          if varargin{2} < 2 && not(varargin{2} == -1) ,
            error('Duration must be at least 2ms (or -1 for Inf)');
          end
          
          dur_code = zeros(1,3, int_type);

          if varargin{2} == -1,
            dur_code = ones(1,3, int_type)*intmax(int_type);
          else
            stim_dur = varargin{2};
            dur_code(3) = floor(stim_dur/100);
            dur_code(2) = floor(mod(stim_dur, 100)/10);
            dur_code(1) = mod(stim_dur, 10);
            if max(dur_code) < 2,
              maxId = find(dur_code, 1,'last');
              dur_code(maxId-1) = dur_code(maxId-1) + 10;
              dur_code(maxId) = dur_code(maxId) -1;
            end
          end

          disp(sprintf('Duration set to %i msec', varargin{2}));
          message = zeros(1,64,int_type);
          message(1:3) = dur_code;
          pnet(socke, 'write', message);
          pnet(socke, 'writepacket');
          
        case 'close'
          pnet(socke, 'close');
          socke= [];    
    end

end