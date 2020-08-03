function mrk_eo= mrk_evenOdd(mrk, className)
%mrk= mrk_evenOdd(mrk, <className>)
%mrk= mrk_evenOdd(mrk, <class index>)


if exist('className','var'), 
  mrk= mrk_selectClasses(mrk, className);
end

mrk_eo= mrk;
nClasses= size(mrk.y,1);
if ~isfield(mrk, 'className'),
  if nClasses>1,
    mrk.className= cprintf('class %d', 1:nClasses)';
  else
    mrk_eo.className= {'even', 'odd'};
  end
end

mrk_eo.y= [mrk.y; zeros(size(mrk.y))];
for c= 1:nClasses,
  idx= find(mrk.y(c,:));
  mrk_eo.y(c,idx(1:2:end))= 0;
  mrk_eo.y(nClasses+c,idx(1:2:end))= 1;
  if isfield(mrk, 'className'),
    mrk_eo.className{c}= [mrk.className{c} ' (even)'];
    mrk_eo.className{nClasses+c}= [mrk.className{c} ' (odd)'];
  end
end
mrk_eo= mrk_selectClasses(mrk_eo, [1:2:2*nClasses, 2:2:2*nClasses]);
