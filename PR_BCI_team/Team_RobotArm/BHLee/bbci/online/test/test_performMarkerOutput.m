opt = struct('fs',100,'log',0);



adminMarker('init',opt);
mrkOutDef(1).marker = -5;
mrkOutDef(1).value = 1;
mrkOutDef(2).marker = -7;
mrkOutDef(2).value = 17;
mrkOutDef(1).no_marker = -1;
mrkOutDef(2).no_marker = -1;


performMarkerOutput('init',opt,[])
mrkOutDef = performMarkerOutput('init',opt,mrkOutDef)


mrkOut = performMarkerOutput('apply',mrkOutDef,40,40)


adminMarker('add',10000,[-30,-10],[-5,7]);

mrkOut = performMarkerOutput('apply',mrkOutDef,40,10040);
mrkOut{:}
adminMarker('add',10040,[],[]);

mrkOut = performMarkerOutput('apply',mrkOutDef,40,10080);
mrkOut{:}





adminMarker('add',10900,[-30 -10],[-7 37]);
mrkOut = performMarkerOutput('apply',mrkOutDef,200,10900);
mrkOut{:}



adminMarker('add',10980,[],[]);
mrkOut = performMarkerOutput('apply',mrkOutDef,80,10980);
mrkOut{:}





