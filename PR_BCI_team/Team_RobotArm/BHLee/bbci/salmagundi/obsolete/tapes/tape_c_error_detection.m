model= 'FisherDiscriminant';

model= 'MSM1';

model= struct('classy', {{'RDA', 1}});
model.param= struct('index',3, 'scale','lin', 'value',[1e-6 .001 .1]); %%QDA

model= struct('classy', {{'RDA', 0}});
model.param= struct('index',3, 'scale','lin', 'value',[1e-6 .001 .1]); %%LDA

model= struct('classy', 'FDqwqx');
model.param= struct('index',2, 'scale','log', 'value',-2:4);
 
model= struct('classy', 'LinSVM');
model.param= struct('index',2, 'scale','log', 'value',-2:4);
