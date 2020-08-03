function [ltype] = icmessage_getlabeltype(name)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETLABELTYPE Returns the numerical id of a label type
%
% LTYPE = ICMESSAGE_GETLABELTYPE(NAME) Returns the numerical id (i.e.
% 0) for the label type in NAME (i.e. 'biosig').
%
% See also ICMESSAGE_GETLABELS
					ltype = -1;
	mex_id_ = 'getltype(i cstring, io int*)';
[ltype] = tobiic(mex_id_, name, ltype);

