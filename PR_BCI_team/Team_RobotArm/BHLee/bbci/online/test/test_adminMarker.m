opt = struct('fs',100,'log',0);



adminMarker('init',opt);

[toe,ts] = adminMarker('query',[-1000 0])



adminMarker('add',10000,[-30,-10],[-5,7]);

[toe,ts] = adminMarker('query',[-1000 -20])

adminMarker('add',10040,[],[]);

[toe,ts] = adminMarker('query',[-1000 -20])

adminMarker('add',10920,[-30,-10],[3,-27]);

adminMarker('add',10980,[],[]);

[toe,ts] = adminMarker('query',[-1000 -20]);

