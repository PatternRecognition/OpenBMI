function [status] = icmessage_addclass(q, name, label, value)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_ADDCLASS Adds an ICClass to an ICClassifier
%
% STATUS = ICMESSAGE_ADDCLASS(Q, NAME, LABEL, VALUE) Adds a new
% ICClass to the ICClassifier called NAME in the ICMessage Q.
% NAME is the classifier name (i.e. "cnbi_mi").
% VALUE is the value (i.e. 0.75 for 'prob')
% LTYPE is the label (i.e. '0x781' for 'biosig')
% Returns 1 on success, 0 and prints an error otherwise.
%
% See also ICMESSAGE_GETVALUES, ICMESSAGE_GETLABELS, ICMESSAGE_ADDCLASSIFIER
													mex_id_ = 'o std::string* = new(i cstring)';
[sname] = tobiic(mex_id_, name);
	mex_id_ = 'o std::string* = new(i cstring)';
[slabel] = tobiic(mex_id_, label);
	mex_id_ = 'o bool = addk(i ICMessage*, i std::string*, i std::string*, i float*)';
[status] = tobiic(mex_id_, q, sname, slabel, value);

