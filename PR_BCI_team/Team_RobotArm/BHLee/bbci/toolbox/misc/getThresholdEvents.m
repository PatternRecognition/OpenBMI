function mrk= getThresholdEvents(trigSignal, trigDef, fs, blockingTime,btSame)
%mrk= getThresholdEvents(trigSignal, trigDef, fs, blockingTime, btSame)

if ~exist('blockingTime','var') | isempty(blockingTime),
  bl= 1;
else
  bl= max(1, blockingTime*fs/1000);
end
if ~exist('btSame','var') | isempty(btSame),
  bs= bl;
else
  bs= max(1, btSame*fs/1000);
end
T= size(trigSignal,1);

nClasses= size(trigDef, 2);
belongsto= zeros(T, 1);
for ic= 1:nClasses,
  thresh= trigDef{2,ic};
%  tch= chanind(cnt, trigDef{1,ic});
%  fi= find(cnt.x(:,tch)>thresh(1) & cnt.x(:,tch)<thresh(2));
  tch= trigDef{1,ic};
  fi= find(trigSignal(:,tch)>=thresh(1) & trigSignal(:,tch)<thresh(2));
  belongsto(fi)= ic;
end
noChange= find(diff([0; belongsto])==0);
belongsto(noChange)= 0;

te= 0;
tt= min(find(belongsto));
mrk.pos= [];
while ~isempty(tt),
  te= te+1;
  mrk.pos(te)= tt;
  while tt-mrk.pos(te)<bs & belongsto(tt)==belongsto(mrk.pos(te)),
    tt= tt+bl-1 + min(find(belongsto(tt+bl:end)));
  end
end
mrk.toe= belongsto(mrk.pos);
mrk.fs= fs;

if te==0,
  warning('threshold criterium never satisfied');
  return;
end

mrk.y= zeros(nClasses, length(mrk.pos));
for ic= 1:nClasses,
  mrk.y(ic,:)= (mrk.toe==ic);
end
if size(trigDef,1)>2,
  mrk.className= {trigDef{3,:}};
end
