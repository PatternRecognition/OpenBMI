setup_augcog;

fil = 6:11;
feature = {'spectrum','bandEnergy','csp','rc','ar','peak_spectrum','bandPowerbyvariance','idealBrain'};
task = {'*auditory','*calc', '*visual', '*comb'};
ival = {[1000 30000],[1000 10000],[1000 5000]};

freq = {[7 15],[4 20]};
spatial = {{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{2,false}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}}};
params = {{{}},{{}},{{''},{'proc_normalizeChannels'}},{{6,6}},{{6}},{{[2 35]}},{{}},{{}}};

channels = {{'not','E*','M*'}}; 

setup = {'lbo','lno','rch','dbb'};
%model_RLDA = struct('classy','RLDA','msDepth', 3,'param',struct('index',2,'value',[0,0.000001,0.0001,0.001,0.1,0.4]));

for fi = 1:length(fil)
  clear test stan;
  fid = fopen(sprintf('/home/tensor/dornhege/calcData/augcog_classification_%s.txt',augcog(fil(fi)).file),'a');
  load(sprintf('/home/tensor/dornhege/calcData/augcog_classification_new2_%s',augcog(fil(fi)).file));
  for ta = 1:length(task)
    blk = getAugCogBlocks(augcog(fil(fi)).file);
    blk = blk_selectBlocks(blk,task{ta});
    if ~isempty(blk.y)
    [cnt,mrk] = readBlocks(blk);
   
    for fe = 1:8
      
      for iv = 1:length(ival)
        for fr = 1:length(freq)
          for sp = 1:length(spatial{fe})
            for pa = 1:length(params{fe})
              for ch = 1:length(channels)
                fprintf(fid,'Choosing subject: %s\n',augcog(fil(fi)).file);
                fprintf(fid,'Choosing task: %s\n',task{ta});
                fprintf(fid,'Choosing feature: %s\n',feature{fe});
                fprintf(fid,'Choosing step-width: %i and Window-length: %i\n',ival{iv});
                fprintf(fid,'Choosing Frequency-band: [%i %i]\n',freq{fr});
                fprintf(fid,'Choosing Spatial Filter: ');
                fprintf(fid,'%s ',spatial{fe}{sp}{:});
                fprintf(fid,'\n');
                fprintf(fid,'Choosing Parameter: ');
                if length(params{fe}{pa})>0 & isnumeric(params{fe}{pa}{1})
                  fprintf(fid,'%f ',params{fe}{pa}{:});
                else
                  fprintf(fid,'%s ',params{fe}{pa}{:});
                end
                fprintf(fid,'\n');
                fprintf(fid,'Choosing Channels: ');
                fprintf(fid,'%s ',channels{ch}{:});
                fprintf(fid,'\n');

                fprintf('Choosing subject: %s\n',augcog(fil(fi)).file);
                fprintf('Choosing task: %s\n',task{ta});
                fprintf('Choosing feature: %s\n',feature{fe});
                fprintf('Choosing step-width: %i and Window-length: %i\n',ival{iv});
                fprintf('Choosing Frequency-band: [%i %i]\n',freq{fr});
                fprintf('Choosing Spatial Filter: ');
                fprintf('%s ',spatial{fe}{sp}{:});
                fprintf('\n');
                fprintf('Choosing Parameter: ');
                if length(params{fe}{pa})>0 & isnumeric(params{fe}{pa}{1})
                  fprintf('%f ',params{fe}{pa}{:});
                else
                  fprintf('%s ',params{fe}{pa}{:});
                end
                fprintf('\n');
                fprintf('Choosing Channels: ');
                fprintf('%s ',channels{ch}{:});
                fprintf('\n');
                
                clear fv;
                fv = feval(['get_augcog_' feature{fe}],cnt,mrk,ival{iv},freq{fr},spatial{fe}{sp},params{fe}{pa}{:},channels{ch});
                
                % get feature
                for se = 1:length(setup)
                  clear divTr divTe;
                  switch setup{se}
                   case 'lbo'
                    fprintf(fid,'Do leave-one-block-out, ');
                    for i =1:length(fv.taskname)
                      divTe{i} = {i};
                      divTr{i} = {[1:i-1,i+1:length(fv.taskname)]};
                    end
                   case 'lno'
                    fprintf(fid,'Do leave-all-neighbors-out, ');
                    for i =1:length(fv.taskname)
                      divTe{i} = {i};
                      divTr{i} = {[1:i-2,i+2:length(fv.taskname)]};
                    end
                   case 'rch'
                    fprintf(fid,'Do leave-two-blocks-out, ');
                    po = 1;
                    for i = 1:2:length(fv.taskname);
                      for j = 2:2:length(fv.taskname);
                        divTe{po} = {[i,j]};
                        divTr{po} = {setdiff(1:length(fv.taskname),[i,j])};
                        po = po+1;
                      end
                    end
                   case 'dbb'
                    fprintf(fid,'Do leave-two-neighbor-blocks-out, ');
                    for i = 1:length(fv.taskname)-1;
                      divTe{i} = {[i,i+1]};
                      divTr{i} = {[1:i-1,i+2:length(fv.taskname)]};
                    end
                  end
                  
                  fv.divTr = divTr; fv.divTe = divTe;
                  [te,st] = xvalidation(fv,'LDA');
                  fprintf(fid,'result LDA: %2.1f +/- %1.1f\n',100*te,100*st);
                  test(ta,fe,iv,fr,sp,pa,ch,se) = te;
                  stan(ta,fe,iv,fr,sp,pa,ch,se) = st;
                  
                  %                  [te,st] = xvalidation(fv,model_RLDA,struct('xTrials',[1 1],'msTrials',[1 1]));
%                 fprintf(fid,'result RLDA: %2.1f +/- %1.1f\n',100*te,100*st);
                end
                fprintf(fid,'\n\n');
              end
            end
          end
        end
      end
    end
  end
  end
  fclose(fid);
  save(sprintf('/home/tensor/dornhege/calcData/augcog_classification_new3_%s',augcog(fil(fi)).file),'test','stan');
end

