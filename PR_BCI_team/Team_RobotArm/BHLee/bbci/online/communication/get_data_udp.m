function handle = get_data_udp(varargin);
%GET_DATA_UDP gets data via udp
%
% usage:
%  <init:>  handle = get_data_udp(port);
%  <get:>   data = get_data_udp(handle,timeout,actual);
%  <close:> get_data_udp(<handle>);
%
% input: 
%   port    a port number (positive)
%   handle: handle given back in the init case (usually a negative number)
%   actual: flag if all packages are got
%   timeout:timeout in secs
% 
% output:
%   handle:  negative number pointing to a open connection
%   data:    received data (empty if nothing is achieved within timeout
%  
% Note: get_data_udp closes all open connections
%
% Guido Dornhege, 07/02/2006
% $Id: get_data_udp.m,v 1.2 2006/07/20 12:18:03 neuro_cvs Exp $


global BBCI_VERBOSE
persistent handles handlelink

if nargin==0
  typ = 0; kill = handles;
end

if nargin==1
  if varargin{1}(1)>100
    typ = 1; ports = varargin{1};
  else
    typ = 0; kill = varargin{1};
  end
end

if nargin==2
  typ = 2; get = varargin{1}; timeout = varargin{2}; actual = 0;
end
if nargin==3
  typ = 2; get = varargin{1}; timeout = varargin{2}; actual = varargin{3};
end

switch typ
 case 1
  host = get_hostname;
  
  handle = setdiff(1:length(handles)+length(ports),handles);
  handle = handle(1:length(ports));
  if any(handle)>64
    error('only 64 handles allowed');
  end
  
  for i = 1:length(ports)
    handlelink{handle(i)} = get_from_udp(host,ports(i));
  end
  handles = union(handles,handle);
  
  if BBCI_VERBOSE,
    fprintf('[get_data_udp] init on port %s:\n', vec2str(ports));
    fprintf('[get_data_udp] handle -> handlelink:\n', vec2str(handle));
    for i= 1:length(handle),
      fprintf('   %d -> %d\n', handle(i), handlelink{handle(i)});
    end
  end
  
 case 2
  if length(get)>1
    error('can only get the results of one communication');
  end
  
  handle = get_from_udp(handlelink{get},timeout);
  if BBCI_VERBOSE,
    if ~isempty(handle),
      fprintf('[get_data_udp] received: %s:\n', vec2str(handle));
    end
  end
  if actual
    aa = handle;
    while ~isempty(aa)
      handle = aa;               %% ???
      handle = get_from_udp(handlelink{get},0);
      if BBCI_VERBOSE,
        if ~iseqmpty(handle),
          fprintf('[get_data_udp] received2: %s:\n', vec2str(handle));
        end
      end
    end
  end
  
 case 0
  
  if BBCI_VERBOSE,
    fprintf('[get_data_udp] closing.\n');
  end
  for i = 1:length(kill)
    get_from_udp(handlelink{kill(i)},'close');
  end
  handles = setdiff(handles,kill);
  
end

