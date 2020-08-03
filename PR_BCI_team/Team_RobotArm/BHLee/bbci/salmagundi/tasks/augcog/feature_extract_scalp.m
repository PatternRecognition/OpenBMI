setup_augcog;
clab = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
        'F7', 'F8', 'T7', 'T8', 'Fz', 'Cz', 'Pz', ...
        'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', ...
        'TP9', 'TP10'};
mnt= projectElectrodePositions(clab);
mnt= setDisplayMontage(mnt, 'augcog');


fil = 6:12;
task = {'*auditory','*calc', '*visual', '*carfollow', '*comb'};
band = [3,7;7 14; 15 30];
bandnames = {'theta','alpha','beta'};

for fi = 1:length(fil);
  for ta = 1:length(task);
    clear fv1 fv2;
    for ba = 1:size(band,1)
      try
        fv = get_augcog_bandEnergy(augcog(fil(fi)).file,task{ta},[2000 20000],band(ba,:),'car',{'not','E*','M*','Fp*'});

        if length(unique(fv.bidx))==8
          fv1{ba} = fv;
          clear divTe divTr
          for i = 1:length(fv.taskname)-1;
            divTe{i} = {[i,i+1]};
            divTr{i} = {[1:i-1,i+2:length(fv.taskname)]};
          end
          
          fv.divTr = divTr; fv.divTe = divTe;
          
          
          model = {'LPM',10000};
          
          [te,st] = xvalidation(fv,model);
          te = 100*te; st = 100*st;
          
          fv.title = sprintf('low-high, %2.1f +/- %1.1f',te,st);
          
        else
          fv.title = sprintf('low-high');
        end
        C = trainClassifier(fv,model);
        
        subplot(3,2,2*ba-1);
        plotScalpPattern(mnt,abs(C.w),struct('scalePos','none'));
        h = text(-1.25,0,bandnames{ba});
        set(h,'rotation',90)
        set(h,'HorizontalAlignment','center')
        title(fv.title);
      end
      try
        fv = get_augcog_bandEnergy(augcog(fil(fi)).file,{['high ' task{ta}(2:end)],'low drive'},[2000 20000],band(ba,:),'car',{'not','E*','M*','Fp*'});
      end
      
      try% there are too much elements in fv
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
      
      fv2{ba} = fv;
      
      clear divTe divTr
      for i = 2:length(fv.taskname)-1;
        divTe{2*i-3} = {[1,i]};
        divTe{2*i-2} = {[length(fv.taskname),i]};
        divTr{2*i-3} = {[2:i-1,i+1:length(fv.taskname)]};
        divTr{2*i-2} = {[1:i-1,i+1:length(fv.taskname)-1]};
      end
      
      if exist('divTr','var')
        fv.divTr = divTr; fv.divTe = divTe;
        
        
        model = {'LPM',10000};
        
        C = trainClassifier(fv,model);
        
        [te,st] = xvalidation(fv,model);
        te = 100*te; st = 100*st;
        fv.title = sprintf('base-high, %2.1f +/-%1.1f',te,st);
      else
        fv.title = sprintf('base-high');
      end
      subplot(3,2,2*ba);
      plotScalpPattern(mnt,abs(C.w),struct('scalePos','none'));
      h = text(-1.25,0,bandnames{ba});
      set(h,'rotation',90)
      set(h,'HorizontalAlignment','center')
      title(fv.title);
      end
    end
    
%    hh = addTitle(sprintf('%s, %s, single band classification',texi(augcog(fil(fi)).file),task{ta}(2:end)));
%    set(hh,'FontSize',0.5*get(hh,'FontSize'));
    saveFigure(['augcog_misc/feature_extraction_LPM_band_' augcog(fil(fi)).file '_' task{ta}(2:end)],[20 30]);
 
  
try    
    fv = fv1{1};
    fv.x = cat(1,fv1{1}.x,fv1{2}.x,fv1{3}.x);
    
    
    clear divTe divTr
    for i = 1:length(fv.taskname)-1;
      divTe{i} = {[i,i+1]};
      divTr{i} = {[1:i-1,i+2:length(fv.taskname)]};
    end
    
    fv.divTr = divTr; fv.divTe = divTe;
    
    if length(unique(fv.bidx))==8
      model = {'LPM',10000};
      
      [te,st] = xvalidation(fv,model);
      te = 100*te; st = 100*st;
      
      fv.title = sprintf('low-high, %2.1f +/- %1.1f',te,st);
    else
      fv.title = sprintf('low-high');
    end
    
      
    C = trainClassifier(fv,model);
    
    ww = reshape(C.w,size(fv.x,1),size(fv.x,2));
    ww = abs(ww);
    wmax = max(ww(:)); wmin = min(ww(:));
    clf;
    for ii = 1:size(ww,1)
      subplot(size(ww,1),2,2*ii-1);
      plotScalpPattern(mnt,abs(ww(ii,:)),struct('scalePos','none','colAx',[wmin,wmax]));
      h = text(-1.25,0,bandnames{ii});
      set(h,'rotation',90)
      set(h,'HorizontalAlignment','center')
      if ii ==1 
        title(fv.title);
      end
    end

    fv = fv2{1};
    fv.x = cat(1,fv2{1}.x,fv2{2}.x,fv2{3}.x);
    
    
    clear divTe divTr
    for i = 2:length(fv.taskname)-1;
      divTe{2*i-3} = {[1,i]};
      divTe{2*i-2} = {[length(fv.taskname),i]};
      divTr{2*i-3} = {[2:i-1,i+1:length(fv.taskname)]};
      divTr{2*i-2} = {[1:i-1,i+1:length(fv.taskname)-1]};
    end
      
    if exist('divTr','var')
      fv.divTr = divTr; fv.divTe = divTe;
      
      model = {'LPM',10000};
      
      [te,st] = xvalidation(fv,model);
      te = 100*te; st = 100*st;
      
      fv.title = sprintf('base-high, %2.1f +/- %1.1f',te,st);
    else
      fv.title = sprintf('base-high');
    end
    
      
    C = trainClassifier(fv,model);
    
    ww = reshape(C.w,size(fv.x,1),size(fv.x,2));
    ww = abs(ww);
    wmax = max(ww(:)); wmin = min(ww(:));
     for ii = 1:size(ww,1)
      subplot(size(ww,1),2,2*ii);
      plotScalpPattern(mnt,abs(ww(ii,:)),struct('scalePos','none','colAx',[wmin,wmax]));
      h = text(-1.25,0,bandnames{ii});
      set(h,'rotation',90)
      set(h,'HorizontalAlignment','center')
      if ii ==1 
        title(fv.title);
      else
        title(' ');
      end
    end

    
%    hh = addTitle(sprintf('%s, %s, all band classification',texi(augcog(fil(fi)).file),task{ta}(2:end)));
%    set(hh,'FontSize',0.5*get(hh,'FontSize'));
    saveFigure(['augcog_misc/feature_extraction_LPM_allband_' augcog(fil(fi)).file '_' task{ta}(2:end)],[20 30]);
 
end    
    
  end
end
