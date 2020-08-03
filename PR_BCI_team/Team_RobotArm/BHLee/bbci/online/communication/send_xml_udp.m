function send_xml_udp(varargin)
%SEND_XML_UDP - Send Signal in XML Format via UDP
%
%Synopsis:
% send_xml_udp('init', hostname, port)
% send_xml_udp('PARAM1', VALUE1, ...)
% send_xml_udp('SIGNAL_TYPE', 'PARAM1', VALUE1, ...)
% send_xml_udp('close')
%
%Arguments:
%  hostname: String, IP-address or hostname
%  port: Integer value, number of the port
%  'PARAM': String, name of the variable to be sent. The default type is
%     'f' (float). Other types can be specified before a colon separator ':'
%     like 'i:integer_variable'.
%  VALUE: Value of the type specified in 'PARAM'.
%  'SIGNAL_TYPE': Type of signal, default 'control-signal'.
%
%Returns:
%  nothing
%
%Example:
% send_xml_udp('init', bbci.fb_machine, bbci.fb_port);
% send_xml_udp('i:controlnumber', controlnumber, ...
%              'timestamp', timestamp, ...
%              'cl_output', classifier_output)

% blanker               

global BCI_DIR
persistent socke

if nargin==3 & isequal(varargin{1},'init'),
  hostname= varargin{2};
  port= varargin{3};
  path(path, [BCI_DIR 'import/tcp_udp_ip']);
  socke= pnet('udpsocket', 1111);  %% what is this port number?
  if socke==-1,
    error('udp communication failed');
  end
  pnet(socke, 'udpconnect', hostname, port);
elseif nargin==1 & isequal(varargin{1}, 'close'),
  pnet(socke, 'close');
  socke= [];
else
  signal_type= 'control-signal';
  if mod(nargin,2)==1,
    signal_type= varargin{1};
    varargin= varargin(2:end);
  end
  if ~all(apply_cellwise2(varargin(1:2:end), 'ischar')) | ~ischar(signal_type),
    error('unrecognized format of input arguments');
  end
  if isempty(socke),
    error('open a udp connection first');
  end
  xml_cmd= ['<?xml version="1.0" ?><bci-signal version="1.0">' ...
            '<' signal_type '>'];
  nVars= nargin/2;
  for ii= 1:nVars,
    var_name= varargin{2*ii-1};
    var_value= varargin{2*ii};
    fmt= 'f';
    is= find(var_name==':', 1, 'first');
    if ~isempty(is),
      fmt= var_name(1:is-1);
      var_name= var_name(is+1:end);
    end
    if length(var_value)<=1 | fmt=='s',
      xml_cmd= strcat(xml_cmd, sprintf(['<%s name="%s" value="%' fmt '" />'],...
                                       fmt, var_name, var_value));
    else
      xml_cmd= strcat(xml_cmd, sprintf('<list name="%s">', var_name));
      for ij= 1:length(var_value),
        xml_cmd= strcat(xml_cmd, sprintf(['<%s value="%' fmt '" />'], fmt, var_value(ij)));
      end
      xml_cmd= strcat(xml_cmd, '</list>');
    end
  end
  xml_cmd= strcat(xml_cmd, ['</' signal_type '></bci-signal>']);
  pnet(socke, 'write', xml_cmd);
  pnet(socke, 'writepacket');
end
