fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4'); 
fv= proc_selectIval(fv, [0 ep]); 
fv= proc_subsampleByMean(fv, 5); 

step= 50; 
erp_cz= proc_selectChannels(epo, 'FCz'); 
erp_cz= proc_movingAverage(erp_cz, 20); 
erp_cz= proc_selectIval(erp_cz, [-100 200]); 
mmh= proc_classDifference(erp_cz, [2 1]); 
mmh= proc_baseline(mmh, [-100 -60]); 
[mi, iNep]= min(mmh.x); 
iNes= min(find(mmh.x(1:iNep)<0.15*mi)); 
iNee= max(find(mmh.x(1:iNep+10)<0.15*mi)); 
NeIval= mmh.t([iNes iNee]), 
inter= fliplr(ep:-step:mmh.t(iNee+1)); 
region= cell(1, length(inter)+1); 
region{1}= NeIval; 
region{2}= [NeIval(2) inter(1)]; 
for is= 2:length(inter), 
  region{is+1}= [inter(is-1)+10 inter(is)]; 
end, 
fv= proc_selectChannels(epo, 'F#', 'FC3-4', 'C3-4', 'CP3-4', 'P3','P4'); 
fv= proc_meansOfRegions(fv, region); 

