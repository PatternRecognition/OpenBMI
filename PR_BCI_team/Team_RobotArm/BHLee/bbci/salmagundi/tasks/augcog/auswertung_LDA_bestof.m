setup_augcog;

fil = 6:12;
feature = {'spectrum','bandEnergy','csp','rc','ar','peak_spectrum','bandVariance','idealBrain'};
task = {'*auditory','*calc', '*visual', '*comb'};
ival = [30000,10000,5000];

freq = {[7 15],[4 20]};
spatial = {{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{2,false}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}}};

setup = {'lbo','lno','rch','dbb'};


fid = fopen('/home/tensor/dornhege/neuro_cvs/Tex/bci/augcog_misc/results_LDA_bestof.tex','w');
fprintf(fid,'\\documentclass[10pt]{article}\n');
fprintf(fid,'\\usepackage{geometry}\n');
fprintf(fid,'\\usepackage[T1]{fontenc}\n');
fprintf(fid,'\\usepackage[german,english]{babel}\n');
fprintf(fid,'\\usepackage{latexsym,amsmath,amssymb,amsfonts,bbm,mathptmx}\n');
fprintf(fid,'\\usepackage[dvips]{graphicx}\n');
fprintf(fid,'\\usepackage[small]{caption}\n');
fprintf(fid,'\\usepackage{units,nicefrac,xspace,longtable,fancyheadings}\n');
fprintf(fid,'\\newcommand{\\std}[1]{{$\\scriptstyle\\pm\\;#1$}}\n');

fprintf(fid,'\\setlength{\\parindent}{0mm}\n');
fprintf(fid,'\\title{Results of LDA classification, best of feature extraction}\n');
fprintf(fid,'\\author{Guido Dornhege}\n');
fprintf(fid,'\\pagestyle{fancyplain}\n');
fprintf(fid,'\\lhead{}\n');
fprintf(fid,'\\rhead{}\n');
fprintf(fid,'\\cfoot{\\rm\\thepage}\n');


fprintf(fid,'\\begin{document}\n');
fprintf(fid,'\\maketitle\n\n');

fprintf(fid,'For all Augcog Experiments of session II (March/April2004) LDA performance was measured under the following setups\n');

fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item Task:\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item auditory\n');
fprintf(fid,'\\item calc\n');
fprintf(fid,'\\item visual (not available for VPth and VPts) \n');
fprintf(fid,'\\item comb (not available for VPts)\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item Feature extraction:\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item spectrum\n');
fprintf(fid,'\\item band Energy\n');
fprintf(fid,'\\item Common spatial pattern csp (2 Pattern without overfitting, optional with normalization)\n');
fprintf(fid,'\\item reflection coefficients rc (6,6)\n');
fprintf(fid,'\\item autoregressive coefficients ar (6)\n');
fprintf(fid,'\\item band energy about estimated 1/f spectrum (peak\_spectrum) (base spectrum: [2 35]) \n');
fprintf(fid,'\\item band power estimated by variance in causal band pass filtered signals\n');
fprintf(fid,'\\item ideal Brain occipital and frontal alpha/theta activity\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item Frequency filter (fr):\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item [7 15]\n');
fprintf(fid,'\\item [4 20]\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item Window length in msec (wl):\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item 30000\n');
fprintf(fid,'\\item 10000\n');
fprintf(fid,'\\item 5000\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item Step width: 1000 msec\n');
fprintf(fid,'\\item Spatial filter except for CSP (sp):\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item commonAverageReference (car)\n');
fprintf(fid,'\\item diagonal laplace filter complete (lap)\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item channels: not E* M*\n');
fprintf(fid,'\\item Evaluation strategy:');
fprintf(fid,'leave-two-neighbored-blocks-out (dbb)\n');
fprintf(fid,'\\item All values are classification error in \\% high vs low with standard deviation since there are different runs of train and test sets\n');
fprintf(fid,'\\item Classification with Linear Discriminant Analysis (simple Baseline-Classifier)\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\clearpage\n');

testins = {}; stanins = {};
for fi = 1:length(fil);
  if exist(['/home/tensor/dornhege/calcData/augcog_classification_new3_',augcog(fil(fi)).file '.mat'],'file')
    
    S = load(['/home/tensor/dornhege/calcData/augcog_classification_new3_',augcog(fil(fi)).file '.mat']);
    test = S.test*100; stan = S.stan*100;
    testins{fi} = test;
    stanins{fi} = stan;
  end
end

fprintf(fid,'\\part{BEST OF SUBJECT AND TASK-SPECIFIC}\n');
anz = 35;

for fi = 1:length(fil);
  if fi<=length(testins)
    fprintf(fid,'\\rhead{$%s$}\n',augcog(fil(fi)).file);
    fprintf(fid,'\\section{$%s$}\n',augcog(fil(fi)).file);
  for ta = 1:length(task);
    if ta<=size(testins{fi},1)
    te = squeeze(testins{fi}(ta,:,:,:,:,:,:,4));
    st = squeeze(stanins{fi}(ta,:,:,:,:,:,:,4));
    t = te([1,2],:,:,:,1);
    t1 = te(3,:,:,1,:);
    t1 = t1(:,:,:,:);
    t2 = te([4:8],:,:,:,1);
    t = cat(1,t,t1,t2);
    t = t(:);
    s = st([1,2],:,:,:,1);
    s1 = st(3,:,:,1,:);
    s1 = s1(:,:,:,:);
    s2 = st([4:8],:,:,:,1);
    s = cat(1,s,s1,s2);
    s = s(:);
    if sum(t>0)>0
      fprintf(fid,'\\subsection{%s}\n',task{ta});
      [t,ind] = sort(t);
      s = s(ind);
      fprintf(fid,'\\begin{tabular}{cccccc}\n');
      fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
      for j = 1:anz
        in = ind(j);
        [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
        if fe==3
          if sp==1
            spat = 'unnormalized';
          else
            spat = 'normalized';
          end
        else
          spat = spatial{1}{sp}{1};
        end
        fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
      end
      fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
    end
    end
  end
end

    
end

                
fprintf(fid,'\\part{Subjectspecific}\n');
for fi = 1:length(fil);
  if fi<=length(testins)
    fprintf(fid,'\\rhead{$%s$}\n',augcog(fil(fi)).file);
    fprintf(fid,'\\section{$%s$}\n',augcog(fil(fi)).file);
    te = testins{fi}(:,:,:,:,:,:,:,4); te = find(sum(te(:,:),2)>0);
    
    st = squeeze(std(testins{fi}(te,:,:,:,:,:,:,4),1));
    te = squeeze(mean(testins{fi}(te,:,:,:,:,:,:,4),1));
    t = te([1,2],:,:,:,1);
    t1 = te(3,:,:,1,:);
    t1 = t1(:,:,:,:);
    t2 = te([4:8],:,:,:,1);
    t = cat(1,t,t1,t2);
    t = t(:);
    s = st([1,2],:,:,:,1);
    s1 = st(3,:,:,1,:);
    s1 = s1(:,:,:,:);
    s2 = st([4:8],:,:,:,1);
    s = cat(1,s,s1,s2);
    s = s(:);
    if sum(t>0)>0
      [t,ind] = sort(t);
      s = s(ind);
      fprintf(fid,'\\begin{tabular}{cccccc}\n');
      fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
      for j = 1:anz
        in = ind(j);
        [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
        if fe==3
          if sp==1
            spat = 'unnormalized';
          else
            spat = 'normalized';
          end
        else
          spat = spatial{1}{sp}{1};
        end
        fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
      end
      fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
    end
    end
    
end


fprintf(fid,'\\part{Subjectspecific without visual}\n');
for fi = 1:length(fil);
  if fi<=length(testins)
    fprintf(fid,'\\rhead{$%s$}\n',augcog(fil(fi)).file);
    fprintf(fid,'\\section{$%s$}\n',augcog(fil(fi)).file);
    te = testins{fi}(:,:,:,:,:,:,:,4); te = find(sum(te(:,:),2)>0);
    te = setdiff(te,3);
    st = squeeze(std(testins{fi}(te,:,:,:,:,:,:,4),1));
    te = squeeze(mean(testins{fi}(te,:,:,:,:,:,:,4),1));
    t = te([1,2],:,:,:,1);
    t1 = te(3,:,:,1,:);
    t1 = t1(:,:,:,:);
    t2 = te([4:8],:,:,:,1);
    t = cat(1,t,t1,t2);
    t = t(:);
    s = st([1,2],:,:,:,1);
    s1 = st(3,:,:,1,:);
    s1 = s1(:,:,:,:);
    s2 = st([4:8],:,:,:,1);
    s = cat(1,s,s1,s2);
    s = s(:);
    if sum(t>0)>0
      [t,ind] = sort(t);
      s = s(ind);
      fprintf(fid,'\\begin{tabular}{cccccc}\n');
      fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
      for j = 1:anz
        in = ind(j);
        [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
        if fe==3
          if sp==1
            spat = 'unnormalized';
          else
            spat = 'normalized';
          end
        else
          spat = spatial{1}{sp}{1};
        end
        fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
      end
      fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
    end
    end
    
end

fprintf(fid,'\\part{Taskspecific}\n');
for ta = 1:length(task);
  fprintf(fid,'\\rhead{$%s$}\n',task{ta});
  fprintf(fid,'\\section{$%s$}\n',task{ta});
  te = [];
  for fi = 1:length(fil);
    if fi<=length(testins)
      if ta<=size(testins{fi},1) 
        tt = testins{fi}(ta,:,:,:,:,:,:,4); tt = sum(tt(:)>0)>0;
        if tt
          te = cat(1,te,testins{fi}(ta,:,:,:,:,:,:,4));
        end
      end
    end
  end
  
  st = squeeze(std(te,[],1));
  te = squeeze(mean(te,1));
    t = te([1,2],:,:,:,1);
    t1 = te(3,:,:,1,:);
    t1 = t1(:,:,:,:);
    t2 = te([4:8],:,:,:,1);
    t = cat(1,t,t1,t2);
    t = t(:);
    s = st([1,2],:,:,:,1);
    s1 = st(3,:,:,1,:);
    s1 = s1(:,:,:,:);
    s2 = st([4:8],:,:,:,1);
    s = cat(1,s,s1,s2);
    s = s(:);
  if sum(t>0)>0
    [t,ind] = sort(t);
    s = s(ind);
    fprintf(fid,'\\begin{tabular}{cccccc}\n');
    fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
    for j = 1:anz
      in = ind(j);
      [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
      if fe==3
        if sp==1
          spat = 'unnormalized';
        else
          spat = 'normalized';
        end
      else
        spat = spatial{1}{sp}{1};
      end
      fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
    end
    fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
  end
end

fprintf(fid,'\\part{Taskspecific without jf}\n');
for ta = 1:length(task);
  fprintf(fid,'\\rhead{$%s$}\n',task{ta});
  fprintf(fid,'\\section{$%s$}\n',task{ta});
  te = [];
  for fi = setdiff(1:length(fil),2);
    if fi<=length(testins)
      if ta<=size(testins{fi},1) 
        tt = testins{fi}(ta,:,:,:,:,:,:,4); tt = sum(tt(:)>0)>0;
        if tt
          te = cat(1,te,testins{fi}(ta,:,:,:,:,:,:,4));
        end
      end
    end
  end

  st = squeeze(std(te,[],1));
  te = squeeze(mean(te,1));
    t = te([1,2],:,:,:,1);
    t1 = te(3,:,:,1,:);
    t1 = t1(:,:,:,:);
    t2 = te([4:8],:,:,:,1);
    t = cat(1,t,t1,t2);
    t = t(:);
    s = st([1,2],:,:,:,1);
    s1 = st(3,:,:,1,:);
    s1 = s1(:,:,:,:);
    s2 = st([4:8],:,:,:,1);
    s = cat(1,s,s1,s2);
    s = s(:);
  if sum(t>0)>0
    [t,ind] = sort(t);
    s = s(ind);
    fprintf(fid,'\\begin{tabular}{cccccc}\n');
    fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
    for j = 1:anz
      in = ind(j);
      [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
      if fe==3
        if sp==1
          spat = 'unnormalized';
        else
          spat = 'normalized';
        end
      else
        spat = spatial{1}{sp}{1};
      end
      fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
    end
    fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
  end
end


fprintf(fid,'\\part{over all}\n');


te = [];
for ta = 1:length(task);
  for fi = 1:length(fil);
    if fi<=length(testins)
      if ta<=size(testins{fi},1)
       tt = testins{fi}(ta,:,:,:,:,:,:,4); tt = sum(tt(:)>0)>0;
        if tt
          te = cat(1,te,testins{fi}(ta,:,:,:,:,:,:,4));
        end
      end
    end
  end
end

st = squeeze(std(te,[],1));
te = squeeze(mean(te,1));
t = te([1,2],:,:,:,1);
t1 = te(3,:,:,1,:);
t1 = t1(:,:,:,:);
t2 = te([4:8],:,:,:,1);
t = cat(1,t,t1,t2);
t = t(:);
s = st([1,2],:,:,:,1);
s1 = st(3,:,:,1,:);
s1 = s1(:,:,:,:);
s2 = st([4:8],:,:,:,1);
s = cat(1,s,s1,s2);
s = s(:);
if sum(t>0)>0
  [t,ind] = sort(t);
  s = s(ind);
  fprintf(fid,'\\begin{tabular}{cccccc}\n');
  fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
  for j = 1:anz
    in = ind(j);
    [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
    if fe==3
      if sp==1
        spat = 'unnormalized';
      else
        spat = 'normalized';
      end
    else
      spat = spatial{1}{sp}{1};
    end
    fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
  end
  fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
end



fprintf(fid,'\\part{over all without jf and visual}\n');

te = [];
for ta = setdiff(1:length(task),3);
  for fi = setdiff(1:length(fil),2);
    if fi<=length(testins)
      if ta<=size(testins{fi},1)
       tt = testins{fi}(ta,:,:,:,:,:,:,4); tt = sum(tt(:)>0)>0;
        if tt
          te = cat(1,te,testins{fi}(ta,:,:,:,:,:,:,4));
        end
      end
    end
  end
end



st = squeeze(std(te,[],1));
te = squeeze(mean(te,1));
t = te([1,2],:,:,:,1);
t1 = te(3,:,:,1,:);
t1 = t1(:,:,:,:);
t2 = te([4:8],:,:,:,1);
t = cat(1,t,t1,t2);
t = t(:);
s = st([1,2],:,:,:,1);
s1 = st(3,:,:,1,:);
s1 = s1(:,:,:,:);
s2 = st([4:8],:,:,:,1);
s = cat(1,s,s1,s2);
s = s(:);
if sum(t>0)>0
  [t,ind] = sort(t);
  s = s(ind);
  fprintf(fid,'\\begin{tabular}{cccccc}\n');
  fprintf(fid,'Pos & Feature & Window length & Frequency & Spatial/Norm & Results\\\\\\hline\n');
  for j = 1:anz
    in = ind(j);
    [fe,iv,fr,sp] = ind2sub([length(feature),length(ival),length(freq),2],in);
    if fe==3
      if sp==1
        spat = 'unnormalized';
      else
        spat = 'normalized';
      end
    else
      spat = spatial{1}{sp}{1};
    end
    fprintf(fid,'%i & $%s$ & %i & [%i %i] & %s & %2.1f\\std{%1.1f} \\\\\n',j,feature{fe},ival(iv),freq{fr}(1),freq{fr}(2),spat,t(j),s(j));
  end
  fprintf(fid,'\\end{tabular}\n\\clearpage\n\n');
end




                
                
   

fprintf(fid,'\\end{document}\n');
fclose(fid);