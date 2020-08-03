function [status] = icmessage_setvalue(q, name, label, value)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_SETVALUE Sets the value of a specific ICClass
%
% STATUS = ICMESSAGE_SETVALUE(Q, NAME, LABEL, VALUE) where Q is an ICMessage. 
% Sets the value of ICClass with LABEL (i.e. '0x781' for 'biosig') belonging to 
% ICClassifier NAME (i.e. 'cnbi_mi') to VALUE (i.e. '0.75' for 'prob').
% Returns 0 on error, 1 upon success.
% 
% See also ICMESSAGE_GETVALUE, ICMESSAGE_ADDCLASS, ICMESSAGE_ADDCLASSIFIER,
% ICMESSAGE_NEW
											mex_id_ = 'o std::string* = new(i cstring)';
[sname] = tobiic(mex_id_, name);
	mex_id_ = 'o std::string* = new(i cstring)';
[slabel] = tobiic(mex_id_, label);
	mex_id_ = 'o bool = setv(i ICMessage*, i std::string*, i std::string*, i float*)';
[status] = tobiic(mex_id_, q, sname, slabel, value);

