setup_augcog;


fil = 6:11;
task = {'*auditory','*calc', '*visual', '*carfollow', '*comb'};


for fi = 1:length(fil);
  for ta = 1:length(task);
    try
      fv = get_augcog_spectrum(augcog(fil(fi)).file,task{ta},[2000 20000],[3 20],'car',{'not','E*','M*','Fp*'});

    
    for i = 1:length(fv.taskname)-1;
      divTe{i} = {[i,i+1]};
      divTr{i} = {[1:i-1,i+2:length(fv.taskname)]};
    end
    
    fv.divTr = divTr; fv.divTe = divTe;

    
    model = {'LPM',10000};
    
    [te,st] = xvalidation(fv,model);
    te = 100*te; st = 100*st;

    fv.title = sprintf('%s %s, low-high, %2.1f +/-%1.1f',augcog(fil(fi)).file,task{ta}(2:end),te,st);
    
    catch
      fv.title = sprintf('%s %s, low-high',augcog(fil(fi)).file,task{ta}(2:end));
    end
    
    C = trainClassifier(fv,model);

    plot_classifierImage(C,fv);

    
    saveFigure(['augcog_misc/feature_extraction_LPM_' augcog(fil(fi)).file '_' task{ta}(2:end) '_low-high'],[25 15]);
  
  
    try
      fv = get_augcog_spectrum(augcog(fil(fi)).file,{['high ' task{ta}(2:end)],'low drive'},[2000 20000],[3 20],'car',{'not','E*','M*','Fp*'});
    end

    % there are too much elements in fv
    el = sum(fv.y,2);
    ge = length(getClassIndices(fv.taskname,'high*'));
    al = round(2*el(2)/ge);
    ind = find(fv.task(1,:));
    ind = ind(randperm(length(ind)));
    ind = ind(al+1:end);
    ind2 = find(fv.task(end,:));
    ind2 = ind2(randperm(length(ind2)));
    ind2 = ind2(al+1:end);
    ind = [ind,ind2];
    fv.y(:,ind)=[];
    fv.task(:,ind)=[];
    fv.x(:,:,ind) = [];
    fv.bidx(ind) = [];
    

    for i = 2:length(fv.taskname)-1;
      divTe{2*i-3} = {[1,i]};
      divTe{2*i-2} = {[length(fv.taskname),i]};
      divTr{2*i-3} = {[2:i-1,i+1:length(fv.taskname)]};
      divTr{2*i-2} = {[1:i-1,i+1:length(fv.taskname)-1]};
    end
    
    fv.divTr = divTr; fv.divTe = divTe;

    
    model = {'LPM',10000};
    
    C = trainClassifier(fv,model);
    
    try
      [te,st] = xvalidation(fv,model);
      te = 100*te; st = 100*st;
      fv.title = sprintf('%s %s, base-high, %2.1f +/-%1.1f',augcog(fil(fi)).file,task{ta}(2:end),te,st);
    catch
      fv.title = sprintf('%s %s, base-high',augcog(fil(fi)).file,task{ta}(2:end));
    end
    plot_classifierImage(C,fv);

    
    saveFigure(['augcog_misc/feature_extraction_LPM_' augcog(fil(fi)).file '_' task{ta}(2:end) '_base-high'],[25 15]);
 
  
  
  end
end
