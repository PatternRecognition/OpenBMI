function send_data_udp(varargin)
%SEND_DATA_UDP sends data via udp
% usage
%  <init:>  send_data_udp(machine,port)
%  <send:>  send_data_udp(val)
%  <close:> send_data_udp
%  <init+send+close:>  send_data_udp(machine,port,val)
%
% input:
%  machine: name of a machine (or cell array of machines)
%  port:    port number (or list of port numbers, double array)
%  val:     double array
%
% Guido Dornhege, 07/02/2006
% $Id: send_data_udp.m,v 1.4 2007/01/11 15:18:42 neuro_cvs Exp $

global BBCI_VERBOSE
persistent handle handlelink

if isempty(handlelink)
  handlelink = cell(1,64);
end

switch nargin
 case 2
  machine = varargin{1};
  port = varargin{2};
  if ~iscell(machine)
    machine = {machine};
  end
  if length(port)==1
    port = port*ones(1,length(machine));
  end

  hh = setdiff(1:length(handle)+length(machine),handle);
  hh = hh(1:length(machine));
  if any(hh)>64
    error('only 64 connections allowed');
  end
  for i = 1:length(machine)
    handlelink{hh(i)} = send_to_udp(machine{i},port(i));
  end
  handle = union(handle,hh);
  
  if BBCI_VERBOSE,
    fprintf('[send_data_udp] init: handle -> handlelink:\n', vec2str(handle));
    for i= 1:length(handle),
      fprintf('   %d -> %d\n', handle(i), handlelink{handle(i)});
    end
  end
  
 case 1
  val = varargin{1};
  
  if BBCI_VERBOSE,
    fprintf('[send_data_udp] sending to handles: %s.\n', vec2str(handle));
%    fprintf('[send_data_udp] sending to handlelinks: %s.\n', vec2str(handlelink));
  end
  for i = 1:length(handle)
     send_to_udp(handlelink{handle(i)},val);
  end
  
 case 0
  try
    if BBCI_VERBOSE,
      fprintf('[send_data_udp] closing handles: %s.\n', vec2str(handle));
    end
    for i = 1:length(handle)
      send_to_udp(handlelink{handle(i)},'close');
    end
  end
  
  handle = [];
  
  
 case 3
  machine = varargin{1};
  port = varargin{2};
  val = varargin{3};
  if ~iscell(machine)
    machine = {machine};
  end
  
  if BBCI_VERBOSE,
    fprintf('[send_data_udp] all in one.\n');
    fprintf('[send_data_udp] machine %s on port %d.\n', machine{1}, port(1));
    if ischar(val),
      fprintf('[send_data_udp] val= %s.\n', val);
    else
      fprintf('[send_data_udp] val= %s.\n', vec2str(val));
    end
  end
  if length(port)==1
    port = port*ones(1,length(machine));
  end
  for i = 1:length(machine)
    hh = send_to_udp(machine{i},port(i));
    send_to_udp(hh,val);
    send_to_udp(hh,'close');
  end
end

