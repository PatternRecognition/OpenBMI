file= 'Gabriel_00_09_05/selfpaced2sGabriel';
[cnt,mrk,mnt]= loadProcessedEEG(file);

epo= makeSegments(cnt, mrk, [-1300 0] - 100);
fv= proc_selectChannels(epo, 'C3','C2');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 100);
fv= proc_subsampleByMean(fv, 10);

doXvalidation(fv, 'LDA', [3 10]);
doXvalidation(fv, 'QDA', [3 10]);

xx= squeeze(fv.x)';
left= find(mrk.y(1,:));
right= find(mrk.y(2,:));

plot(xx(left,1), xx(left,2), 'r.'); hold on;
plot(xx(right,1), xx(right,2), 'g.'); 

xlabel('C3');
ylabel('C2');
legend('left events','right events',0);

%% show LDA and QDA separation
CL = train_LDA(xx',fv.y);
CQ = train_QDA(xx',fv.y);


xmin=getfield(get(gca,'XLim'),{1});
ymin=getfield(get(gca,'YLim'),{1});
xmax=getfield(get(gca,'XLim'),{2});
ymax=getfield(get(gca,'YLim'),{2});
for x = 0:100
    xxx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = CL.w'*[xxx;yy] + CL.b;
    end
end

[bla,LmO] = contour(xmin:0.01*(xmax-xmin):xmax, ...
                    ymin:0.01*(ymax-ymin):ymax, v, [0 0], 'c');

for x = 0:100
    xxx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = [xxx,yy]*CQ.sq*[xxx;yy] + CQ.w'*[xxx;yy] + CQ.b;
    end
end

[bla,QmO] = contour(xmin:0.01*(xmax-xmin):xmax, ...
                    ymin:0.01*(ymax-ymin):ymax, v, [0 0], 'm');

%% mark outliers (in covariance distance measure)
le = xx(left,:)*pinv(sqrtm(cov(xx(left,:))));
ri = xx(right,:)*pinv(sqrtm(cov(xx(right,:))));

mle = mean(xx(left,:))*pinv(sqrtm(cov(xx(left,:))));
mri = mean(xx(right,:))*pinv(sqrtm(cov(xx(right,:))));

le = le-repmat(mle,[size(le,1),1]);
ri = ri-repmat(mri,[size(ri,1),1]);

le = sum(le.*le,2)';
ri = sum(ri.*ri,2)';

le1 = sort(le);
ri1 = sort(ri);

pp = figure;
bar(le1);
lef=input('Value:');
bar(ri1);
rig = input('Value:');
close(pp);
outL = left(find(le>lef));
outR = right(find(ri>rig));

plot(xx(outL,1), xx(outL,2), 'rx', 'markerSize',10);
plot(xx(outR,1), xx(outR,2), 'gx', 'markerSize',10);

%% show LDA and QDA separation without outliers
out = union(outL,outR);
xxx = xx;
xxx(out,:) = [];
lab = fv.y;
lab(:,out)=[];
CL = train_LDA(xxx',lab);
CQ = train_QDA(xxx',lab);
for x = 0:100
    xxx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = CL.w'*[xxx;yy] + CL.b;
    end
end

[bla,LoO] = contour(xmin:0.01*(xmax-xmin):xmax, ...
                    ymin:0.01*(ymax-ymin):ymax, v, [0 0], 'y');

for x = 0:100
    xxx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = [xxx,yy]*CQ.sq*[xxx;yy] + CQ.w'*[xxx;yy] + CQ.b;
    end
end

[bla,QoO] = contour(xmin:0.01*(xmax-xmin):xmax, ...
                    ymin:0.01*(ymax-ymin):ymax, v, [0 0], 'b');


legend([LmO,QmO(1),LoO,QoO(1)], 'LDA mit Outlier', 'QDA mit Outlier', ...
       'LDA ohne Outlier', 'QDA ohne Outlier',-1);
hold off;
