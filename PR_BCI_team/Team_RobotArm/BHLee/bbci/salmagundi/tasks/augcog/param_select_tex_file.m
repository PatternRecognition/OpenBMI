show_best = 4;
show_best_2 = 20;

setup_augcog;

S = load('/home/tensor/dornhege/calcData/param_select/all_results');

fid = fopen('/home/tensor/dornhege/neuro_cvs/Tex/bci/augcog_misc/param_select_bestof.tex','w');

fprintf(fid,'\\documentclass[10pt]{article}\n');
fprintf(fid,'\\usepackage[top=15mm,bottom=15mm]{geometry}\n');
fprintf(fid,'\\usepackage[T1]{fontenc}\n');
fprintf(fid,'\\usepackage[german,english]{babel}\n');
fprintf(fid,'\\usepackage{latexsym,amsmath,amssymb,amsfonts,bbm,mathptmx}\n');
fprintf(fid,'\\usepackage[dvips]{graphicx}\n');
fprintf(fid,'\\usepackage[small]{caption}\n');
fprintf(fid,'\\usepackage{units,nicefrac,xspace,longtable,fancyheadings}\n');
fprintf(fid,'\\newcommand{\\std}[1]{{$\\scriptstyle\\pm\\;#1$}}\n');

fprintf(fid,'\\graphicspath{{../pics/augcog_misc/}}\n');
fprintf(fid,'\\setlength{\\parindent}{0mm}\n');
fprintf(fid,'\\title{Results of Parameter Selection in Session 3}\n');
fprintf(fid,'\\author{Guido Dornhege}\n');
fprintf(fid,'\\pagestyle{fancyplain}\n');
fprintf(fid,'\\lhead{}\n');
fprintf(fid,'\\rhead{}\n');
fprintf(fid,'\\cfoot{\\rm\\thepage}\n');


fprintf(fid,'\\begin{document}\n');
fprintf(fid,'\\maketitle\n\n');

fprintf(fid,'\\setlongtables\n');
fprintf(fid,'For all Augcog Experiments of session III (June 2004) classification on BandPower was measured for different parameter setups\n');

fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item Subject\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.fil)
  fprintf(fid,'\\item %s\n',convert_subject(augcog(S.fil(i)).file));
end
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item Task:\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.task)
  fprintf(fid,'\\item %s\n',S.task{i}(2:end));
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Feature\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.feature)
  fprintf(fid,'\\item %s\n',S.feature{i});
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Windowlength training\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.ival)
  fprintf(fid,'\\item %i msec\n',S.ival{i});
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Windowlength apply\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.ival_apply)
  fprintf(fid,'\\item %i msec\n',S.ival_apply{i});
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Frequency\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.freq)
  fprintf(fid,'\\item [%i %i] Hz\n',S.freq{i}(1),S.freq{i}(2));
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Spatial\n');
fprintf(fid,'\\begin{itemize}\n');

spati = {};
for i = 1:length(S.spatial)
  for j = 1:length(S.spatial{i});
    spati = cat(2,spati,{S.spatial{i}{j}{1}});
  end
end
spati = unique(spati);

for i = 1:length(spati)
  fprintf(fid,'\\item %s\n',spati{i});
end
fprintf(fid,'\\end{itemize}\n');


pati = {};
for i = 1:length(S.params)
  for j = 1:length(S.params{i});
    if length(S.params{i}{j})>0
      pati = cat(2,pati,{S.params{i}{j}{1}});
    end
  end
end
pati = unique(pati);

if length(pati)>0
  fprintf(fid,'\\item Further params\n');
  fprintf(fid,'\\begin{itemize}\n');
  for i = 1:length(pati)
    fprintf(fid,'\\item %s\n',pati{i});
  end
  fprintf(fid,'\\end{itemize}\n');
end

fprintf(fid,'\\item Channels\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.channels)
  fprintf(fid,'\\item');
  cha = convert_rhomb(S.channels{i});
  fprintf(fid,' %s',cha{:});
  fprintf(fid,'\n');
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Classes\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.CL)
  fprintf(fid,'\\item %s %s\n',S.CL{i}{1}(1:end-1),S.CL{i}{2}(1:end-1));
end
fprintf(fid,'\\end{itemize}\n');

fprintf(fid,'\\item Model\n');
fprintf(fid,'\\begin{itemize}\n');
for i = 1:length(S.modell)
  modi = S.modell{i};
  if isstruct(modi)
    modi = modi.classy;
  elseif iscell(modi)
    modi = modi{1};
  end
   
  fprintf(fid,'\\item %s\n',modi);
end
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\end{itemize}\n');



fprintf(fid,'\\clearpage\n');



for fi = 1:length(S.fil)
  fprintf(fid,'\\lhead{%s}\n',convert_subject(augcog(S.fil(fi)).file));
  fprintf(fid,'\\section{%s}\n',convert_subject(augcog(S.fil(fi)).file));
  for ta = 1:length(S.task)
    fprintf(fid,'\\rhead{%s}\n',S.task{ta}(2:end));
    fprintf(fid,'\\subsection{%s}\n',S.task{ta}(2:end));
    for cl = 1:length(S.CL)
      fprintf(fid,'\\chead{%s-%s}\n',S.CL{cl}{1}(1:end-1),S.CL{cl}{2}(1:end-1));
      fprintf(fid,'\\subsubsection{%s-%s}\n',S.CL{cl}{1}(1:end-1),S.CL{cl}{2}(1:end-1));
    
      res = S.result(fi,ta,:,:,:,:,:,:,:,cl,:);
      resi = res(:);
      [dum,ind] = sort(-resi);
      re = -dum;
      ii = cell(1,11);
      [ii{:}] = ind2sub(size(res),ind);    
      ii{1} = fi*ii{1};ii{2} = ta*ii{2};ii{10} = cl*ii{10};
      rel = find(size(res)>1);
      fprintf(fid,'\\begin{longtable}{cc}\n');
      stri = {};
      stri2 = {};
      for i = 1:min(show_best_2,length(ind))
        st = ''; st2 = 'param_select_';
        st2 = [st2,augcog(S.fil(ii{1}(i))).file,'_'];
        if ismember(1,rel)
          st = [st,augcog(S.fil(ii{1}(i))).file,' '];
        end
        st2 = [st2,S.CL{ii{10}(i)}{1}(1:end-1),S.CL{ii{10}(i)}{2}(1:end-1),'_'];
        if ismember(10,rel)
          st = [st,S.CL{ii{10}(i)}{1}(1:end-1),' ',S.CL{ii{10}(i)}{2}(1:end-1),' '];
        end
        st2 = [st2,S.task{ii{2}(i)}(2:end),'_'];
        if ismember(2,rel)
          st = [st,S.task{ii{2}(i)}(2:end),' '];
        end
         
        st2 = [st2,S.feature{ii{3}(i)},'_'];
        if ismember(3,rel)
          st = [st,S.feature{ii{3}(i)},' '];
        end
         
        st2 = [st2,num2str(S.ival{ii{4}(i)}),'_'];
        if ismember(4,rel)
          st = [st,'trainival: ', num2str(S.ival{ii{4}(i)}),'msec '];
        end
         
        st2 = [st2,num2str(S.freq{ii{5}(i)}(1)),'_',num2str(S.freq{ii{5}(i)}(2)),'_'];
        if ismember(5,rel)
          st = [st,'Freq: [', num2str(S.freq{ii{5}(i)}(1)),' ', num2str(S.freq{ii{5}(i)}(2)),'] Hz '];
        end
         
        st2 = [st2,S.spatial{ii{3}(i)}{ii{6}(i)}{1},'_'];
        if ismember(6,rel)
          st = [st,S.spatial{ii{3}(i)}{ii{6}(i)}{1},' '];
        end
        
        st2 = [st2,num2str(ii{7}(i)),'_'];
        if ismember(7,rel)
          st = [st,'Param: ',num2str(ii{7}(i)),' '];
        end
        
        
        st2 = [st2,num2str(S.ival_apply{ii{9}(i)}),'_'];
        if ismember(9,rel)
          st = [st,'testival: ', num2str(S.ival_apply{ii{9}(i)}),'msec '];
        end
        
        modi = S.modell{ii{11}(i)};
        if isstruct(modi)
          modi = modi.classy;
        elseif iscell(modi)
          modi = modi{1};
        end
        st2 = [st2,modi,'_'];
        if ismember(11,rel)
          st = [st,'Cl: ', modi,' '];
        end
        

        
        cha = convert_rhomb(S.channels{ii{8}(i)},'>');
        st2 = [st2,sprintf('%s_',cha{:})];
        if ismember(8,rel)
          cha = convert_rhomb(S.channels{ii{8}(i)});
          st = [st,'Channels: ', sprintf('%s ',cha{:})];
        end
        
        st2 = st2(1:end-1); st = st(1:end-1);
        
        stri = cat(1,stri,{st});
        stri2 = cat(1,stri2,{st2});
      end

      stri2 = stri2(1:min(length(stri2),show_best));
      
      for i = 1:2:length(stri2)
        fprintf(fid,'\\parbox{0.45\\linewidth}{%i (%2.1f): %s} &\n',i,100*re(i),stri{i});
        if i<length(stri2)
          fprintf(fid,'\\parbox{0.45\\linewidth}{%i (%2.1f): %s} \\\\[3mm]\n',i+1,100*re(i+1),stri{i+1});
        else
          fprintf(fid,'\\\\\n');
        end
        fprintf(fid,'\\includegraphics[width=0.45\\linewidth]{%s} &\n',stri2{i});
        if i<length(stri2)
          fprintf(fid,'\\includegraphics[width=0.45\\linewidth]{%s} \\\\[5mm]\n',stri2{i+1});
        else
          fprintf(fid,'\\\\\n');
        end
        if mod(i,4)==3 & i+1<length(stri2)
          fprintf(fid,'\\clearpage\n');
        end
      end
      
        
        
      fprintf(fid,'\\end{longtable}\n');
      for i = show_best+1:length(stri)
        fprintf(fid,'%i. (%2.1f): %s\n\n',i,100*re(i),stri{i});
      end
        
      fprintf(fid,'\\clearpage\n');
    end
  end
end

fprintf(fid,'\\end{document}\n');




