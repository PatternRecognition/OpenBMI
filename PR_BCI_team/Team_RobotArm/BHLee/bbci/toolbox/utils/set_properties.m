function opt = set_properties(prop,defaults)
% set_properties - process properties according to the toolbox conventions
%  
% Synopsis:
%   opt = set_properties(prop,defaults)
%
% Arguments:
%   prop      - properties, usually varargin from a calling function.
%   defaults  - a cell array of default values as 'Property'/value pairs
%
% Returns:
%   opt       - a property structure (see propertylist2struct)
%
% Description:
% set_properties handles the three possible way in which properties can
% be specified:
% - if properties are specified as a propertystruct nothing is done,
% - if properties are specified as 'Property'/value pairs they are
% converted to a propertystruct by calling propertylist2struct
% - if properties are empty only defaults are used.
% After set_properties is called it is GUARANTEED that properties are
% encoded as a propertystruct, it is however not guaranteed that all
% properties are set. The latter is the case only if defaults are set for
% all properties in the cell array provided as the second argument to a
% call to set_properties.
%
% $Id: set_properties.m,v 1.2 2005/03/01 08:24:10 neuro_toolbox Exp $
% 
% Copyright (C) 2005 Fraunhofer FIRST
% Author: Pavel Laskov (laskov@first.fhg.de), 

if isempty(prop)
  opt = propertylist2struct(defaults{:});
else
  if ispropertystruct(prop{1})
    opt = prop{1};
  else
    opt = propertylist2struct(prop{:});
  end
  opt = set_defaults(opt,defaults{:});
end

