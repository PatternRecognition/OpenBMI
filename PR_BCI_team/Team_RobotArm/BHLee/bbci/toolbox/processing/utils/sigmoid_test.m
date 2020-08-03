m1 = [0 0];
m2 = [0 2];
m3 = [1 0];

Sig = .4*eye(2);

d = Sig*randn(2,100)+repmat(m1',1,100);
e = Sig*randn(2,100)+repmat(m2',1,100);
f = Sig*randn(2,100)+repmat(m3',1,100);
fv = struct;
fv.x = [d e f];
fv.y = [(1:300)<101;((1:300)>100) == ((1:300)<201);(1:300)>200];
figure(1);clf;subplot(1,2,1);hold on;
cols = {'r','g','b'};
for ii = 1:3
  plot(fv.x(1,find(fv.y(ii,:))),fv.x(2,find(fv.y(ii,:))),['.' cols{ii}]);
end
title('The Data');  


for ii = 1:3
  fv_ii = copyStruct(fv,'y');
  ind = find((1:3)~=ii);
  fv_ii.y(2,:) = fv.y(ii,:);
  fv_ii.y(1,:) = sum(fv.y(ind,:),1);
  C_ii = trainClassifier(fv_ii,'LDA');
  out = applyClassifier(fv_ii,'LDA',C_ii);
  [A,B] = sigmoid_training(out,fv_ii.y);
  subplot(4,2,2*ii);
  plot((sigmoid_function(out,A,B)),cols{ii});
  store_out(ii,:) = sigmoid_function(out,A,B);
  title(['p(y=' cols{ii} ')']);
end
subplot(4,2,8);
hold on;
for ii = 1:3
  ind = find((1:3)~=ii);
  ma_ind = find(store_out(ii,:)>max(store_out(ind(1),:),store_out(ind(2),:)));
  plot(ma_ind,store_out(ii,ma_ind),['x' cols{ii}]);
end
title('Winner takes all');