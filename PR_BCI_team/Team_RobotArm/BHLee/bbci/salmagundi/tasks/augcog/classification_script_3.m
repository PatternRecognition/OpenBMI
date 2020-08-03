setup_augcog;

fil = 18;
feature = {'spectrum','bandEnergy','csp','rc','ar','peak_spectrum','bandPowerbyvariance','idealBrain'};
task = {'*auditory','*calc', '*visual', '*carfollow'};
ival = {[2000 30000],[2000 10000],[2000 5000]};

CL = {{'base*','low*'},{'base*','high*'},{'low*','high*'}};
freq = {[7 15],[4 20]};
spatial = {{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{2,false}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}},{{'car'},{'diagonal','filter all'}}};
params = {{{}},{{}},{{''}},{{6,6}},{{6}},{{[2 35]}},{{}},{{}}};

channels = {{'not','E*','M*'}}; 

setup = {'lro'};
%model_RLDA = struct('classy','RLDA','msDepth', 3,'param',struct('index',2,'value',[0,0.000001,0.0001,0.001,0.1,0.4]));

for fi = 1:length(fil)
  clear test stan;
  fid = fopen(sprintf('/home/tensor/dornhege/calcData/augcog_classification_%s.txt',augcog(fil(fi)).file),'a');
%  load(sprintf('/home/tensor/dornhege/calcData/augcog_classification_new2_%s',augcog(fil(fi)).file));
  for ta = 1:length(task)
    blk = getAugCogBlocks(augcog(fil(fi)).file);
    blk = blk_selectBlocks(blk,task{ta});
    if ~isempty(blk.y)
    [cnt,mrk] = readBlocks(blk);
   
    for fe = 1:length(feature)
      
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
                
                
                fprintf(fid,'Do leave-each-round-out, ');
                mrr = separate_markers(mrk);
                ind = find(mrk.y(1,:));
                blo = zeros(1,size(mrk.y,2));
                blo(ind) = 1;
                blo = cumsum(blo);
                [ii,jj] = find(mrr.y);
                blo = cat(1,blo,ii');
                
                for cl = 1:3
                  classes = CL{cl};
                  fprintf(fid,'Classes: %s-%s, ',classes{:});
                  clear divTr divTe;
                  clInd = getClassIndices(mrk.className,classes);
                  ind = find(sum(mrk.y(clInd,:),1));
                  fvv = proc_selectClasses(fv,classes);
                  bl = blo(:,ind);
                  
                  bid = max(bl(1,:));
                  for i = 1:bid
                    divTe{i} = {unique(bl(2,find(bl(1,:)==i)))};
                    divTr{i} = {unique(bl(2,find(bl(1,:)~=i)))};
                  end
                  
                  
                  
% $$$                   bi = unique(fvv.bidx);
% $$$                   
% $$$                   
% $$$                   bid = unique(bl(2,:));
% $$$  
% $$$                   
% $$$                   
% $$$                   
% $$$                   for i = 1:length(bid)
% $$$                     fvv.bidx(fvv.bidx==bid(i))=i;
% $$$                   end
                  
% $$$                   for i = 1:length(bid)
% $$$                     divTe{i} = {find(bl(2,:)==bid(i))};
% $$$                     divTr{i} = {find(bl(2,:)~=bid(i))};
% $$$                   end
% $$$                   
                  fvv.divTr = divTr; fvv.divTe = divTe;
                  [te,st,out] = xvalidation(fvv,'LDA',struct('out_trainloss',1));
                  fprintf(fid,'result LDA: %2.1f +/- %1.1f\n',100*te,100*st);
                  test(ta,fe,iv,fr,sp,pa,ch,cl) = te(1);
                  train(ta,fe,iv,fr,sp,pa,ch,cl) = te(2);
                  stan(ta,fe,iv,fr,sp,pa,ch,cl) = st(1);
                  stantrain(ta,fe,iv,fr,sp,pa,ch,cl) = st(2);
                  output{ta,fe,iv,fr,sp,pa,ch,cl} = out;
                end
                %                  [te,st] = xvalidation(fv,model_RLDA,struct('xTrials',[1 1],'msTrials',[1 1]));
                %                 fprintf(fid,'result RLDA: %2.1f +/- %1.1f\n',100*te,100*st);
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
  save(sprintf('/home/tensor/dornhege/calcData/augcog_classification_%s',augcog(fil(fi)).file),'test','train','stan','stantrain','output');
end

