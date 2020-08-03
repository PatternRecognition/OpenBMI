model= 'FisherDiscriminant';

model= struct('classy', 'RLSR');
model.param= struct('index',2, 'scale','log', 'value',-2:5);
 
model= struct('classy', 'FDqwqx');
model.param= struct('index',2, 'scale','log', 'value',-2:4);
 
model= struct('classy', 'FDlwqx');
model.param= struct('index',2, 'scale','log', 'value',0:5);
 
model= struct('classy', 'LPM');
model.param= struct('index',2, 'scale','log', 'value',-2:3);
 
model= struct('classy', 'LinSVM');
model.param= struct('index',2, 'scale','log', 'value',-2:4);
