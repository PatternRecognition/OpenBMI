function send_udp_xml(varargin)
%SEND_UDP_XML - Send Signal in XML Format via UDP
%
%Synopsis:
% send_udp_xml('init', HOSTNAME, PORT)
% send_udp_xml('PARAM1', VALUE1, ...)
% send_udp_xml('SIGNAL_TYPE', 'PARAM1', VALUE1, ...)
% send_udp_xml('close')
%
%Arguments:
%  HOSTNAME: String, IP-address or hostname
%  PORT: Integer value, number of the port
%  'PARAM': String, name of the variable to be sent. The default type is
%     'f' (float). Other types can be specified before a colon separator ':'
%     like 'i:integer_variable'.
%  VALUE: Value of the type specified in 'PARAM'.
%     N-dimensional matrices are transmitted as nested python lists.
%  'SIGNAL_TYPE': Type of signal, default 'control-signal'.
%
%Returns:
%  nothing
%
%
%Example:
% send_udp_xml('init', bbci.fb_machine, bbci.fb_port);
% send_udp_xml('i:controlnumber', controlnumber, ...
%              'timestamp', timestamp, ...
%              'cl_output', classifier_output)

% blanker, martijn


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
    if strcmp(var_name, 'command'),
      xml_cmd= strcat(xml_cmd, sprintf('<command value="%s"/>', var_value));
    elseif strcmp(var_name, 'savevariables') || strcmp(var_name, 'loadvariables'),
      xml_cmd= strcat(xml_cmd, sprintf('<command value="%s"> <dict> <tuple> <s value="filename"/> <u value="%s"/> </tuple> </dict> </command>', ...
                                       var_name, var_value));
    else
      fmt= 'f';
      is= find(var_name==':', 1, 'first');
      if ~isempty(is),
        fmt= var_name(1:is-1);
        var_name= var_name(is+1:end);
      end
      if iscell(var_value),
        xml_cmd= strcat(xml_cmd, nested_lists(var_value, fmt, var_name));
      elseif length(var_value)<=1 | fmt=='s',
        xml_cmd= strcat(xml_cmd, sprintf(['<%s name="%s" value="%' fmt '" />'],...
                                         fmt, var_name, var_value));
      elseif fmt=='b',
        xml_cmd= strcat(xml_cmd, sprintf(['<b name="%s" value="%s" />'],...
                                         var_name, var_value));
      elseif ndims(var_value) > 1 && length(find(size(var_value) ~= 1)) >= 2,
          %% multidimensional matrix, nested listings needed. Do recursively.
          idx = [1:ndims(var_value)];
          idx(1) = 2; idx(2) = 1;
          xml_cmd = strcat(xml_cmd, expand_matrix(permute(var_value, idx), fmt, var_name));
      else
        xml_cmd= strcat(xml_cmd, sprintf('<list name="%s">', var_name));
        for ij= 1:length(var_value),
          xml_cmd= strcat(xml_cmd, sprintf(['<%s value="%' fmt '" />'], fmt, var_value(ij)));
        end
        xml_cmd= strcat(xml_cmd, '</list>');
      end
    end
  end
  xml_cmd= strcat(xml_cmd, ['</' signal_type '></bci-signal>']);
  pnet(socke, 'write', xml_cmd);
  pnet(socke, 'writepacket');
end



function xml_cmd= nested_lists(var_value, fmt, var_name)
% fmt is just used for numeric entries
xml_cmd = '';
if iscell(var_value) || (numel(var_value)>1 && ~ischar(var_value)),
  if isempty(var_name),
    xml_cmd= strcat(xml_cmd, '<list>');
  else
    xml_cmd= strcat(xml_cmd, sprintf('<list name="%s">', var_name));
  end
end

if iscell(var_value),
  if numel(var_value)>length(var_value),
    error('Only flat cell arrays allowed');
  end
  for k= 1:numel(var_value),
    xml_cmd= strcat(xml_cmd, nested_lists(var_value{k}, fmt, ''));
  end
else
  if isnumeric(var_value) && numel(var_value)>1,
     warning('experimental');
     xml_cmd= strcat(xml_cmd, expand_matrix(var_value, fmt, ''));
    for ij= 1:length(var_value),
      xml_cmd= strcat(xml_cmd, sprintf(['<%s value="%' fmt '" />'], fmt, var_value(ij)));
    end
  elseif ischar(var_value),
    xml_cmd= strcat(xml_cmd, sprintf(['<s value="%s" />'], var_value));
  elseif isnumeric(var_value),
    xml_cmd= strcat(xml_cmd, sprintf(['<%s value="%' fmt '" />'], fmt, var_value));
  else
    error('unexpected type');
  end
end
  
if iscell(var_value) || (numel(var_value)>1 && ~ischar(var_value)),
  xml_cmd = strcat(xml_cmd, '</list>');
end




function xml_cmd = expand_matrix(var_value, fmt, var_name)

xml_cmd = '';
if ~isempty(var_name),
    xml_cmd= strcat(xml_cmd, sprintf('<list name="%s">', var_name));
end
sz = size(var_value);
nd = ndims(var_value);

if nd == 2 && sz(2) == 1,
    for ij= 1:length(var_value),
      xml_cmd= strcat(xml_cmd, sprintf(['<%s value="%' fmt '" />'], fmt, var_value(ij)));
    end
else
    pIdx = [length(sz) 1:length(sz)-1];
    new_sz = sz(1:end-1);
    if length(new_sz) == 1,
        new_sz(2) = 1;
    end
    var_value = permute(var_value, pIdx);
    for ij = 1:sz(end),
        xml_cmd = strcat(xml_cmd, '<list>');
        xml_cmd = strcat(xml_cmd, expand_matrix(reshape(squeeze(var_value(ij,:)), new_sz), fmt, ''));
        xml_cmd = strcat(xml_cmd, '</list>');
    end
end    

if ~isempty(var_name),
    xml_cmd = strcat(xml_cmd, '</list>');
end
