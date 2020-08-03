function [vtype] = icmessage_getvaluetype(name)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETVALUETYPE Returns the numerical id of a value type
%
% VTYPE = ICMESSAGE_GETVALUETYPE(NAME) Returns the numerical id (i.e.
% 0) for the value type in NAME (i.e. 'prob').
%
% See also ICMESSAGE_GETVALUES
					vtype = -1;
	mex_id_ = 'getvtype(i cstring, io int*)';
[vtype] = tobiic(mex_id_, name, vtype);

