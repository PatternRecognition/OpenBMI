function dest = merge_fields(sources)

% function dest = merge_fields(sources)
%
% Merge the fields of all structures in the source cell array into a
% single destination structure. If fields with the same name exist
% in more than one array, the last occurrence will be used.
%
% IN:
%    source - cell array of structures
%
% OUT:
%    dest - destination structure
%
% $Id: merge_fields.m,v 1.1 2004/10/13 14:33:11 neuro_toolbox Exp $
% 
% Copyright (C) 2002 Fraunhofer FIRST
% Author: Pavel Laskov (laskov@first.fhg.de)


for i=1:length(sources)
  fields = fieldnames(sources{i});
  for j=1:length(fields)
    cmd = sprintf('dest.%s = sources{%d}.%s;',fields{j},i,fields{j});
    eval(cmd);
  end
end
