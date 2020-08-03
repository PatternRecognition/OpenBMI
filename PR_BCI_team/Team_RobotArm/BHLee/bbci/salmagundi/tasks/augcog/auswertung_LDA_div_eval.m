setup_augcog;

fil = 6:11;
feature = {'spectrum','bandEnergy','csp','rc','ar','peak_spectrum'};
task = {'*auditory','*calc', '*visual', '*comb'};
ival = [30000,10000,5000];

freq = {[7 15],[4 20]};
spatial = {{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{2,false}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}}};

setup = {'lbo','lno','rch','dbb'};


fid = fopen('/home/tensor/dornhege/neuro_cvs/Tex/bci/augcog_misc/results_LDA_div_eval.tex','w');
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
fprintf(fid,'\\title{Results of LDA classification, different evaluation strategies}\n');
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
fprintf(fid,'\\item Evaluation strategies:\n');
fprintf(fid,'\\begin{itemize}\n');
fprintf(fid,'\\item leave-one-block-out (lbo)\n');
fprintf(fid,'\\item leave-all-neighbors-(of the test set)-out (lno)\n');
fprintf(fid,'\\item choose as test set all combinations of one high and low block (rch)\n');
fprintf(fid,'\\item leave-two-neighbored-blocks-out (dbb)\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\item All values are classification error in \\% high vs low with standard deviation since there are different runs of train and test sets\n');
fprintf(fid,'\\item Classification with Linear Discriminant Analysis (simple Baseline-Classifier)\n');
fprintf(fid,'\\end{itemize}\n');
fprintf(fid,'\\clearpage\n');


for fi = 1:length(fil);
  if exist(['/home/tensor/dornhege/calcData/augcog_classification_',augcog(fil(fi)).file '.mat'],'file')
    fprintf(fid,'\\rhead{$%s$}\n',augcog(fil(fi)).file);
    fprintf(fid,'\\section{$%s$}\n',augcog(fil(fi)).file);
    S = load(['/home/tensor/dornhege/calcData/augcog_classification_',augcog(fil(fi)).file '.mat']);
    test = S.test*100; stan = S.stan*100;
    for ta = 1:length(task)
      if size(test,1)>=ta & sum(test(ta,:)~=0)>0
        fprintf(fid,'\\lhead{%s}\n',task{ta}(2:end));
        fprintf(fid,'\\subsection{%s}\n',task{ta}(2:end));
        for fe = 1:length(feature);
          fprintf(fid,'\\subsubsection{$%s$}\n',feature{fe});
          fprintf(fid,'\\begin{table}[ht]\n');
          fprintf(fid,'\\begin{tabular}{c|cccc}\n');
          fprintf(fid,'PROC (wl/fr/sp) & lbo & lno & rch & dbb');
          fla = '\hline';
          for iv = 1:length(ival)
            for fr = 1:length(freq)
              switch feature{fe}
               case 'csp'
                fprintf(fid,'\\\\%s\n%i/[%i %i]/%s',fla,ival(iv),freq{fr},'unnorm');fla='';
                te = test(ta,fe,iv,fr,1,1,1,:);
                st = stan(ta,fe,iv,fr,1,1,1,:);
                for i = 1:length(setup)
                  fprintf(fid,' & %2.1f \\std{%1.1f}',te(i),st(i));
                end
                fprintf(fid,'\\\\\n%i/[%i %i]/%s',ival(iv),freq{fr},'norm');
                te = test(ta,fe,iv,fr,1,2,1,:);
                st = stan(ta,fe,iv,fr,1,2,1,:);
                for i = 1:length(setup)
                  fprintf(fid,' & %2.1f \\std{%1.1f}',te(i),st(i));
                end
              
               otherwise
                for sp = 1:length(spatial{fe})
                  te = test(ta,fe,iv,fr,sp,1,1,:);
                  st = stan(ta,fe,iv,fr,sp,1,1,:);
                  spa = spatial{fe}{sp}{1}; if strcmp(spa,'diagonal'), spa='lap';end
                  fprintf(fid,'\\\\%s\n%i/[%i %i]/%s',fla,ival(iv),freq{fr},spa);fla = '';
                  for i = 1:length(setup)
                    fprintf(fid,' & %2.1f \\std{%1.1f}',te(i),st(i));
                  end
                end
              end
            end
          end
          fprintf(fid,'\n\\end{tabular}\n');
          fprintf(fid,'\\caption{Classification results for different evaluation strategies for $%s$, %s and feature $%s$}\n',augcog(fil(fi)).file,task{ta}(2:end),feature{fe});
          fprintf(fid,'\\end{table}\n');
          if mod(fe,2)==0
            fprintf(fid,'\\clearpage\n');
          end
        end
      end
    end
  end
end






fprintf(fid,'\\end{document}\n');
fclose(fid);