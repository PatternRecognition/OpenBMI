fil = 6;
blklength = 60000;
blkdist = 15000;
triallength = 15000;

setup_augcog;


blk = getAugCogBlocks(augcog(fil).file);
blk.ival(2,1)= blk.ival(1,2);

blk = separate_markers(blk);
blk = blk_selectBlocks(blk,'low drive I');

% create blocks
iv = blk.ival;
blklength = blklength*blk.fs/1000;
blkdist = blkdist*blk.fs/1000;

iv = iv(1)+blkdist:blklength+blkdist:iv(2)-blkdist-blklength;

if mod(length(iv),2)==1
  iv(end) = [];
end

blk.ival = [iv;iv+blklength];
blk.className = {'low','high'};

switch 2
 case 1 
  % rand
  yy = randperm(length(iv));
  blk.y = zeros(1,length(iv));
  blk.y(1,yy(yy(1:0.5*length(iv))))= 1;
  blk.y = [blk.y;1-blk.y];
 case 2
  % wechsel
  blk.y = zeros(1,length(iv));
  blk.y(1,1:2:end) = 1;
  blk.y = [blk.y;1-blk.y];
end  

%blk = separate_markers(blk);
%blk = mrk_setMarkers(blk,triallength);

fv = get_augcog_bandPowerbyvariance(blk,'',triallength,[7 15],'car');

model = {'SVM',0.01,'gaussian',1};
gf('send_command','add_preproc NORMONE');

[te_block] = xvalidation(fv,model,[1 1]);

fv2 = fv;
fv2 = rmfield(fv,'bidx');
[te_trial] = xvalidation(fv2,model,[1 1]);

fprintf('Block validation: %2.1f\n',100*te_block);
fprintf('Trial validation: %2.1f\n',100*te_trial);
