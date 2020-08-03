model_LDA= 'LDA';


model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];


model_RDA= struct('classy', 'RDA', ...
                  'msDepth',2, 'inflvar',2);
model_RDA.param(1)= struct('index',2, ...
                           'value',[0 0.05 0.25 0.5 0.75 0.9 1]);
model_RDA.param(2)= struct('index',3, ...
                           'value',[0 0.001 0.01 0.1 0.3 0.5 0.7]);


model_LPM= struct('classy','LPM', 'msDepth',3, 'inflvar',2);
model_LPM.param= struct('index',2, 'scale','log', ...
                        'value', [0 1 10 100]);

model_FDlwlx= struct('classy',{{'FDlwlx','*log'}}, 'msDepth',2);
model_FDlwlx.param= [-1:3];

model_knn= struct('classy','kNearestNeighbor', 'msDepth',1);
model_knn.param= [1 3 5 9 15 23];


gf('send_command', 'set_output /dev/null');
model_SVMlin= struct('classy',{{'SVM','*log','linear'}});
model_SVMlin.msDepth= 2;
model_SVMlin.param.index= 2;
model_SVMlin.param.scale= 'log';
model_SVMlin.param.value= [-2 -1 0 1 2];


model_SVMrbf= struct('classy', {{'SVM','*','gaussian','*'}});
model_SVMrbf.msDepth= 2;
model_SVMrbf.param(1)= struct('index',2, 'scale','log', ...
                              'value',[1:3]);
model_SVMrbf.param(2)= struct('index',4, 'scale','log', ...
                              'value',[-1 -0.5 0 1]);

model_list= struct('name', ...
   {'LDA', 'RLDA', 'RDA', 'LPM', 'kNN', 'linear SVM', 'gaussian SVM'}, ...
   'model', {model_LDA, model_RLDA, model_RDA, model_LPM, model_knn, ...
             model_SVMlin, model_SVMrbf});

opt= struct('xTrials',[5 10]);
%% you should never do this:
opt.outer_ms= 1;

N= load_uci_dataset('');
for mm= 1:length(model_list),
  fprintf('\n\nresults for <%s>\n\n', model_list(mm).name);
  for n= 1:N,
    fv= load_uci_dataset(n);
    [err(n,mm),err_std(n,mm)]= xvalidation(fv, model_list(mm).model, opt, ...
                                           'out_prefix',[fv.title ': ']);
  end  
end

errorbar(err, err_std);
legend(method.name);
