function epo= proc_areaActivation2(epo, cind)
%epo= proc_areaActivation(epo, clab_areas)

sx= cat(2, sum(sum(epo.x(:,cind{1},:),1),2), ...
           sum(sum(epo.x(:,cind{2},:),1),2));
epo.x= sx;
epo= rmfield(epo, 't');

