function bool= str_matchesHead(head, str)
%STR_MATCHESHEAD - Is one string the head of another string?
%
%Synopsis:
% TF= str_matchesHead(HEAD, STR)
%
%Arguments:
%  HEAD:  CHAR - Short string
%  STR :  CHAR - String that is presumably shorter
%
%Description:
%  This function decides whether the string HEAD is the head, i.e.,
%  the first part of the second string STR.
%
%Returns:
%  TF : BOOL - Truth value of HEAD being the head of STR

% 06-2012 Some Genius

bool= strncmp(head, str, length(head));