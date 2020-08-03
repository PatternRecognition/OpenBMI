model= 'FisherDiscriminant',

model= 'QDA',

model= struct('classy', 'RLSR'),
model.param= struct('index',2, 'scale','log', 'value',-1:2:5);

model= struct('classy', 'FDlwqx'),
model.param= struct('index',2, 'scale','log', 'value',0:4);
model.msDepth= [2 5];

model= struct('classy', 'RDA'),
model.param= struct('index',2, 'value',[0 .05 .1 .2 .35 .5 .75 .95 1]);
model.param(2)= struct('index',3, 'value',[0 .25 .5 .75 1]);
