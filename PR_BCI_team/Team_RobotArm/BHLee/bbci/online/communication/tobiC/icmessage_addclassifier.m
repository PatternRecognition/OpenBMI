function [status] = icmessage_addclassifier(q, name, desc, vtype, ltype)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_ADDCLASSIFIER Adds an ICClassifier to an ICMessage
%
% STATUS = ICMESSAGE_ADDCLASSIFIER(Q, NAME, DESC, VTYPE, LTYPE) Adds a new
% ICClassifier to the ICMessage Q. 
% NAME is the name of the classifier (i.e. "cnbi_mi"), DESC the description
% field (i.e. "CNBI MI Classifier").
% VTYPE is the numerical id of the value type (i.e. 0 for probabilities).
% LTYPE is the numerical id of the label type (i.e. 0 for Biosig labels).
% Returns 1 on success, 0 and prints a message on error otherwise.
%
% See also ICMESSAGE_GETVALUES, ICMESSAGE_GETLABELS
												mex_id_ = 'o std::string* = new(i cstring)';
[sname] = tobiic(mex_id_, name);
	mex_id_ = 'o std::string* = new(i cstring)';
[sdesc] = tobiic(mex_id_, desc);
	mex_id_ = 'o bool = addc(i ICMessage*, i std::string*, i std::string*, i int*, i int*)';
[status] = tobiic(mex_id_, q, sname, sdesc, vtype, ltype);

