
function ppcode = vidname2ppcode(name)
   if regexp(name,'.*LQ0.*')==1
     ppcode = 100;
   elseif regexp(name,'.*LQ1.*')==1
     ppcode = 101;
   elseif regexp(name,'.*LQ2.*')==1
     ppcode = 102;
   elseif regexp(name,'.*LQ3.*')==1
     ppcode = 103;
   elseif regexp(name,'.*HQ_only.*')==1
     ppcode = 104;
   else
     error('Filename does not match any pp_code.')
   end