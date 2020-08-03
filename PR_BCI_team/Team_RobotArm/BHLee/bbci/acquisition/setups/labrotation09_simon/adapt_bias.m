
pos_bound = 0.25;
bootstrap = 1;
disp(['Adapting classifier bias to result in a cls_kp probability of ' num2str(1-pos_bound) ' ... '])

% create nokp-markers (by bootstrapping) such that the ratio kp:nokp = pos_bound:(1-pos_bound)
feat = analyze.features;
if bootstrap
kp = find(feat.y(1,:));
nokp = find(feat.y(2,:));
nNewMarkers = (3*length(kp)-length(nokp));
if nNewMarkers~=0
  newnokp.y = zeros(2,nNewMarkers);
  newnokp.x = zeros(size(feat.x,1),size(feat.x,2),nNewMarkers);
  newnokp.className = feat.className;
  for n = 1:nNewMarkers
    idx = nokp(ceil(rand*length(nokp)));
    newnokp.y(:,n) = feat.y(:,idx);
    newnokp.x(:,:,n) = feat.x(:,:,idx);
  end
  feat = proc_appendEpochs(feat, newnokp);
end
end

% choose bias such that cls_output is negative (i.e. cls_kp) in 'pos_bound' of the cases
frac = floor(size(feat.y,2)*pos_bound);
cls_out = cls.C.w' * proc_flaten(feat.x);
[so,si]= sort(cls_out);
old_bias = cls.C.b;
cls.C.b = -(so(frac)+eps);
disp(['Finished (old bias: ' num2str(old_bias,3) ', new bias: ' num2str(cls.C.b,3) ')'])
