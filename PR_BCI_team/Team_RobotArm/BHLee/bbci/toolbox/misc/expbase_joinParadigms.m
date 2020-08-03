function newbase= expbase_joinParadigms(expbase, Para1, Para2)
%expbase= expbase_joinParadigms(expbase, para1, para2)

%% bb ida.first.fhg.de 07/2004

if iscell(Para1),
  para1= Para1{1};
  matchmode1= {Para1{2}};
else
  para1= Para1;
  matchmode1= {};
end

if iscell(Para2),
  para2= Para2{1};
  matchmode2= {Para2{2}};
else
  para2= Para2;
  matchmode2= {};
end

iRemove= [];
newbase= expbase;
idx= strmatch(para1, {expbase.paradigm}, matchmode1{:});
for ii=idx',
  iSameSub= strmatch(expbase(ii).subject, {expbase.subject}, 'exact');
  iSameDat= strmatch(expbase(ii).date, {expbase.date}, 'exact');
  iSameExp= intersect(iSameSub, iSameDat);
  idx_p2= strmatch(para2, {expbase(iSameExp).paradigm}, matchmode1{:});
  idx_p2= iSameExp(idx_p2);
  if length(idx_p2)>0,
    newbase(ii).paradigm= {expbase([ii; idx_p2]).paradigm};
    iRemove= cat(1, iRemove, idx_p2);
  end
end
newbase(iRemove)= [];
