function [pairs, pace, sy]= getEventPairs(mrk, timeLimit)
%[pairs, pace, sy]= getEventPairs(mrk, timeLimit/paceStr)
%
% only implemented for two classes, i.e., size(mrk.y,1) must be 2.
% groups events according to predecessor. labelling class 1 by 'l' and
% class to by 'r' the output is 
%
% pairs= {ll rl lr rr};
%
% only events are considered with predecessors that happended at most
% >timeLimit< ms before the acutal one. if the second argument is a 
% string (>paceStr<) the function getTimeLimit is called.
%
% C getTimeLimit, equiSubset

if ischar(timeLimit),                 %% 'timeLimit' is paceStr
  timeLimit= getTimeLimit(timeLimit);
end

this= [1 2]*mrk.y;
last= [0, this(1:end-1)];
ds= diff(mrk.pos*1000/mrk.fs);
inter= [0, ds];  
sy= ds(ds<timeLimit);
ll= find(last==1 & this==1 & inter<timeLimit);
rl= find(last==2 & this==1 & inter<timeLimit);
lr= find(last==1 & this==2 & inter<timeLimit);
rr= find(last==2 & this==2 & inter<timeLimit);

pairs= {ll rl lr rr};
pace= mean(sy);
