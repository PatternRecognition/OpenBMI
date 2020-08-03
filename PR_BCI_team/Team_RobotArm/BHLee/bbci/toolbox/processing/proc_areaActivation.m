function epo= proc_areaActivation(epo, clab_areas)
%epo= proc_areaActivation(epo, clab_areas)

cind= cell(1, length(clab_areas));
for ca= 1:length(clab_areas),
  if ~iscell(clab_areas{ca}),
    switch(clab_areas{ca}),
     case 'left-hemi',
      cind{ca}= strpatternmatch({'*1','*3','*5','*7','*9'}, epo.clab);
     case 'right-hemi',
      cind{ca}= strpatternmatch({'*0','*2','*4','*6','*8'}, epo.clab);
     otherwise,
      error('unknown keyowrd');
    end
  else
    cind{ca}= strpatternmatch(clab_areas{ca}, epo.clab);
    clab_areas{ca}= ['area' int2str(ca)];
  end
end

sx= cat(2, sum(sum(epo.x(:,cind{1},:),1),2), ...
           sum(sum(epo.x(:,cind{2},:),1),2));
epo.x= sx;
epo= rmfield(epo, 't');
epo.clab= clab_areas;
