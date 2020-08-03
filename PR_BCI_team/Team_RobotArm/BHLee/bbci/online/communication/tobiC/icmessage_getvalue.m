function [retval] = icmessage_getvalue(q, name, label)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETVALUE Returns the value of a specific ICClass
%
% [RETVAL, STATUS] = ICMESSAGE_GETVALUE(Q, NAME, LABEL) where Q is an ICMessage. 
% Returns the value RETVAL of ICClass with LABEL (i.e. '0x781' for 'biosig')
% belonging to ICClassifier NAME (i.e. 'cnbi_mi').
% STATUS is set to NaN if an error occurs.
%
% See also ICMESSAGE_SETVALUE, ICMESSAGE_ADDCLASS, ICMESSAGE_ADDCLASSIFIER,
% ICMESSAGE_NEW
									retval = NaN;
	mex_id_ = 'o std::string* = new(i cstring)';
[sname] = tobiic(mex_id_, name);
	mex_id_ = 'o std::string* = new(i cstring)';
[slabel] = tobiic(mex_id_, label);
	mex_id_ = 'getv(i ICMessage*, i std::string*, i std::string*, io float*)';
[retval] = tobiic(mex_id_, q, sname, slabel, retval);

