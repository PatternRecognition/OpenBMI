clear cont_proc feature cls post_proc marker_output 
global general_port_fields

switch feedback
 case 'csp_1d'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(1).proc = {'online_filtafterDerivation'};
  [b,a] = butter(5,[7 30]/50);
  cont_proc(1).proc_param = {{b,a,randn(21,4)}};

  feature(1).cnt = 1;
  feature(1).ilen_apply = 1000;
  feature(1).proc = {'proc_variance','proc_logarithm'};
  feature(1).proc_param = {{},{}};

  
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(4,1));
  cls(1).integrate = 8;
  cls(1).dist = 0.1;

 case 'csp_1d_3classes'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(1).proc = {'online_filtafterDerivation'};
  [b,a] = butter(5,[7 30]/50);
  cont_proc(1).proc_param = {{b,a,randn(21,6)}};

  feature(1).cnt = 1;
  feature(1).ilen_apply = 1000;
  feature(1).proc = {'proc_variance','proc_logarithm'};
  feature(1).proc_param = {{},{}};

  
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn(3,1),'w',randn(6,3));
  cls(1).integrate = 8;
  cls(1).bias = [2 1 2]';
  
  post_proc.proc = 'three2onePostProc';
  post_proc.proc_param = {};
  
    
 case 'csp_2d'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(1).proc = {'online_filtafterDerivation'};
  [b,a] = butter(5,[7 30]/50);
  cont_proc(1).proc_param = {{b,a,randn(21,4)}};
  cont_proc(2).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(2).proc = {'online_filtafterDerivation'};
  [b,a] = butter(5,[7 30]/50);
  cont_proc(2).proc_param = {{b,a,randn(21,2)}};
  
  feature(1).cnt = 1;
  feature(1).ilen_apply = 1000;
  feature(1).proc = {'proc_variance','proc_logarithm'};
  feature(1).proc_param = {{},{}};
  feature(2).cnt = 2;
  feature(2).ilen_apply= 500;
  feature(2).proc = {'proc_variance','proc_logarithm'};
  feature(2).proc_param = {{},{}};
  
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(4,1));
  cls(1).integrate = 8;
  cls(1).dist = 0.1;

  cls(2).fv = 2;
  cls(2).applyFcn = 'apply_separatingHyperplane';
  cls(2).C = struct('b',randn,'w',randn(2,1));
  cls(2).bias = 5;
  cls(2).alpha = 2;
  
  
 
 case 'brainpong'
  player = 2;
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(1).proc = {'online_filtafterDerivation'};
  [b,a] = butter(5,[7 30]/50);
  cont_proc(1).proc_param = {{b,a,randn(21,4)}};
  
  feature(1).cnt = 1;
  feature(1).ilen_apply = 1000;
  feature(1).proc = {'proc_variance','proc_logarithm'};
  feature(1).proc_param = {{},{}};
  
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(4,1));
  cls(1).scale = 2;
  
  post_proc.proc = 'add_constants';
  post_proc.proc_param = {[1],[player]};
  
 case 'selfpaced'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};

  feature(1).cnt = 1;
  feature(1).ilen_apply = 1300;
  feature(1).proc = {'proc_filtBruteFFT','proc_jumpingMeans'};
  feature(1).proc_param = {{[0.8 4],128,100},{5}};

  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(42,1));

  cls(2).condition = 'F(cl{1}>0);';
  cls(2).fv = 1;
  cls(2).applyFcn = 'apply_separatingHyperplane';
  cls(2).C = struct('b',randn,'w',randn(42,1));
  
  
 case 'pace'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  cont_proc(1).proc = {'online_subtractMovingAverage'};
  cont_proc(1).proc_param = {{1000,'causal'}};
  
  feature(1).cnt = 1;
  feature(1).ilen_apply = 200;
  feature(1).proc = {'proc_jumpingMeans'};
  feature(1).proc_param = {{10}};
  
  cls(1).condition = 'M({{1},[100,100]});';
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(42,1));
  
 case 'reactive'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  
  
  feature(1).cnt = 1;
  feature(1).ilen_apply = 800;
  feature(1).proc = {'proc_baseline','proc_selectIval','proc_jumpingMeans'};
  feature(1).proc_param = {{300},{200},{10}};

  cls(1).condition = 'M({{70,74,102,106},[-100,-100]});';
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(42,1));
  
  marker_output.marker = {70,74,102,106};
  marker_output.value = [70,74,102,106];
  marker_output.no_marker = 0;
    
  
 case 'reactive2'
  cont_proc(1).clab = {'C5','C3','C1','Cz','C2','C4','C6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'};
  
  
  feature(1).cnt = 1;
  feature(1).ilen_apply = 800;
  feature(1).proc = {'proc_baseline','proc_selectIval','proc_jumpingMeans'};
  feature(1).proc_param = {{300},{200},{10}};

  cls(1).condition = 'M({{1,2},[200,500]});';
  cls(1).fv = 1;
  cls(1).applyFcn = 'apply_separatingHyperplane';
  cls(1).C = struct('b',randn,'w',randn(42,1));

  cls(2).condition = 'F(cl{1}>0);';
  cls(2).fv = 1;
  cls(2).applyFcn = 'apply_separatingHyperplane';
  cls(2).C = struct('b',randn,'w',randn(42,1));

  marker_output.marker = {70,74,102,106};
  marker_output.value = [70,74,102,106];
  marker_output.no_marker = 0;
    
  
 otherwise
  error('feedback type not known');
end

  

opt = struct('host',general_port_fields(1).bvmachine,'log',0,'fs',100);


