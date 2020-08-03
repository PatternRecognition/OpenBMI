function set_general_port_fields(typ)
%SET_GENERAL_PORT_FIELDS - Set the global variable _general_port_fields
%
%Synopsis:
% set_general_port_fields(TYPE)
%
%Arguements:
% TYPE: 'localhost' or 'hostbyname' or cell array {bvmachine,control,<graphic>}.

global general_port_fields
       
if ischar(typ),
  switch(typ),
    case 'localhost'
      general_port_fields = struct('bvmachine','127.0.0.1',...
                                   'control',{{'127.0.0.1',12471,12489}},...
                                   'graphic',{{'',12470}});
    case 'hostbyname'
      general_port_fields = struct('bvmachine',get_hostname,...
                                   'control',{{get_hostname,12471,12489}},...
                                   'graphic',{{'',12470}});
    otherwise
      error('unrecognized format');
  end
else
  if ~iscell(typ) | length(typ)<2 | length(typ)>3,
    error('unrecognized format');
  end
  general_port_fields = strukt('bvmachine',typ{1},...
                               'control',{typ{2},12471,12489}, ...
                               'graphic', {'', 12470});
  if length(typ)>2,
    general_port_fields.graphic{1}= typ{3};
  end
end
