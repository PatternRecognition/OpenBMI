function [flag,timeshift,mrk_from_condition]= getCondition(condition, condition_param, ...
						    cl, packetLength)
%function [flag,timeshift,mrk_from_condition]= ...
%               getCondition(condition, condition_param, cl, packetLength)
%
% condition: 
%    c -> A(c1,c2) 
%    c -> O(c1,c2) 
%    c -> N(c1) 
%    c ->
%    F(f(cl{1},...,cl{i-1},condition_param{1},...,condition_param{end})>0);
%    c ->
%    F(f(cl{1},...,cl{i-1},condition_param{1},...,condition_param{end})>0)==0);
%    f can be an arbitrary function term.
%    c -> M(markerCond); 
%    markerCond -> {{M1,...,Mn},[i1,i2]}
%    Mi -> '***01*0' (wildcards for the bits which don't matter)
%    Mi -> n         (natural number)
%    i1=i2<0 or 0<=i1<=i2
%
%    if condition is empty or NaN, it evaluates to true.
% condition_param
%    cell array of optional parameters for conditions
% cl
%    previously calculated classifier outputs.
% packetLength
%    size of the last received data block.

% kraulem 2/12/2004
% TODO: extended documentation by Schwaighase
% $Id: getCondition.m,v 1.1 2006/04/27 14:22:08 neuro_cvs Exp $

if isempty(condition)|isnan(condition)
  flag = true;
  timeshift = [];
else
  [flag,timeshift,condition,mrk_from_condition] = parseCond(condition,...
					 condition_param ,cl, packetLength);
  % if condition is not empty: something went wrong.
  if ~isempty(condition)
    error('Cannot parse Condition.');
  end
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Auxiliary functions: %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [flag,timeshift,condition,mrk_from_condition]= ...
    parseCond(condition, ...
						condition_param,cl, ...
						packetLength)
mrk_from_condition= []; % for 'A','N','O''F' it unclear how to define this
% decide which function should be used.
switch(condition(1))
 case {'A','N','O'}
  % and, not, or: boolean
  [flag,timeshift,condition] = booleanCond(condition, ...
					   condition_param,cl,packetLength);
 case 'M'
  % markerCond
  [flag,timeshift,condition,mrk_from_condition] = markerCond(condition, ...
					  condition_param,cl,packetLength);
 case 'F'
  % function term evaluation
  [flag,timeshift,condition] = functionCond(condition, ...
					    condition_param,cl,packetLength);
end
return


function [flag,timeshift,condition] = functionCond(condition, ...
						  condition_param, ...
						  cl,packetLength)
% evaluate arithmetic conditions.
timeshift = [];
ind = strfind(condition,';');
% error if this can't be found:
if isempty(ind)
  error(sprintf('Missing '';'' in  condition %s',condition));
end
ind = ind(1);
try
  flag = eval(condition(3:(ind-2)));
catch
  error(sprintf('Condition not well-formed: %s',condition));
end
condition = condition((ind+1):end);
return


function [flag,timeshift,condition] = booleanCond(condition, ...
						  condition_param, ...
						  cl,packetLength)
% combine boolean conditions.
junctor = condition(1);
[flag1,timeshift1,condition] = parseCond(condition(3:end), ...
					 condition_param, cl,packetLength);
switch junctor
 case 'A'
  % AND
  [flag2,timeshift2,condition] = parseCond(condition(2:end), ...
					   condition_param, cl,packetLength);
  flag = flag1&flag2;
  timeshift = combineTimeshifts(timeshift1,timeshift2);
 case 'O'
  % OR
  [flag2,timeshift2,condition] = parseCond(condition(2:end), ...
					   condition_param, cl,packetLength);
  flag = flag1|flag2;
  timeshift = combineTimeshifts(timeshift1,timeshift2);
 case 'N'
  % NOT
  flag = ~flag1;
  timeshift = timeshift1;
 otherwise
  error(sprintf('Unknown junctor: %s',junctor));
end
if isempty(condition)
  error('Reached end before finishing parsing Condition.');
end
condition = condition(2:end);
return


function [flag,timeshift,condition,mrk_from_condition] = markerCond(condition, ...
						 condition_param, ...
						 cl,packetLength)
% evaluate a marker condition statement.

ind = strfind(condition,';');
% error if this can't be found:
if isempty(ind)
  error(sprintf('Missing '';'' in  condition %s',condition));
end
ind = ind(1);
try
  mrkArray = eval(condition(3:(ind-2)));
  if length(mrkArray{2})<2
    mrkArray{2} = [mrkArray{2}(1),mrkArray{2}(1)];
  end
catch
  error(sprintf('Condition not well-formed: %s',condition));
end
condition = condition((ind+1):end);

% find the correct interval
if mrkArray{2}(1)>=0
  % find those markers who should be processed in their future
  ival = [min(mrkArray{2}(1)-mrkArray{2}(2),-packetLength),0]-mrkArray{2}(1);
  future = 1;
else
  % find those markers who need a processing with information from the past.
  ival = [max(mrkArray{2}(1),-packetLength) 0];
  future = 0;
end


% query the markers from this interval
[mrkFound,timestamp] = adminMarker('query',ival);
% if isempty(mrkFound),
%   fprintf('no markers in ival [%d %d]\n', ival);
% else
%   fprintf('** Marker found: %d ind ival [%d %d]\n', mrkFound, ival);
% end
% traverse the found markers and decide if they match any of the
% wanted markers.
i = [];
for mr = 1:length(mrkFound)
  i = mrkMatch(mrkFound(mr),mrkArray{1});
  % mrkMatch returns at most one index, otherwise empty.
  if ~isempty(i)
    break;
  end
end
if ~isempty(i) 
  % we found a matching marker. calculate the timeshift!
   flag = true;
   timeshift = timestamp(mr);
   mrk_from_condition= mrkFound(mr);
   if mrkArray{2}(1)>=0
    % information only before the beginning of the timeshift
    % should be taken into account.
    timeshift = timeshift+mrkArray{2}(1);
   end
%   if future==1
%     timeshift = -timeshift
%   end
%     timeshift = timeshift-10;
else
  flag = false;
  timeshift = [];
  mrk_from_condition= [];
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% very specialized auxiliary functions %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		       
		       
function timeshift = combineTimeshifts(timeshift1, timeshift2)
% combining different time shifts in boolean expressions.
if isempty(timeshift1)
  timeshift = timeshift2;
elseif isempty(timeshift2)
  timeshift = timeshift1;
elseif timeshift1==timeshift2
  timeshift = timeshift1;
else
  % two concurring timeshifts yield a problem.
  error('Concurring timeshifts.');
end
return

