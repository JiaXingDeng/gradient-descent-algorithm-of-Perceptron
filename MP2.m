function[wt,t]=MP2(x,y,r,w0)
%%
%使用批量梯度下降法BGD生成单层感知机，其中：
% x指数据,y指对应标签，r指梯度下降学习率，w0为初始值（不能为[0,0,0]'）
% 输出中wt指最终得出的直线参数，t指迭代次数
% EXAMPLE：
% load('perceptron.mat');
%[wt,t]=MP2(x,y,0.9,[1,1,1]')
Y=y;
X=[x',ones(15,1)];%数据加列
WX=X*w0;
bWX= WX > 0;
bY= Y == 1;
inx = bWX ~= Y';%判断数据是否正确
t=0;
error=[];
while sum(inx)>0 & t<10000
    tempx=X(inx,:);
    tempy=Y(inx);
    deltaw = tempx'*tempy';%使用所有错误数据进行迭代
    wt = w0 + r * deltaw;
    w0 = wt;
    t = t + 1;
    bWX=[];
    bY=[];
    inx=[];
    WX=X*w0;
    bWX= WX >= 0;
    bY= Y == 1;
    inx = bWX ~= bY';
    error(t) = -(X(inx,:)*w0/norm(w0))' * Y(inx)';%计算每一步的loss函数
end
%%
%绘图
xp = x(:,y>0);
xn = x(:,y<0);
figure(1)
subplot(221)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid on

axis([0 12 0 8])
set(gca,'position',[0.04 0.55 0.44 0.43]) 
xlabel('(a) Training Data')

hold off  
subplot(222)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid on
p1 = 0;
p2 = (-wt(1)*p1-wt(3))/wt(2);
q1 = 12;
q2 = (-wt(1)*q1-wt(3))/wt(2);
plot([p1 q1],[p2 q2],'k-','linewidth',1.5)
axis([0 12 0 8])
set(gca,'position',[0.5 0.55 0.44 0.43]) 
xlabel('(b) Decision Boundary of Perceptron')

subplot(212)
tt = 1:10:t;
plot(tt,error(tt),'b-','linewidth',1.5);
xlabel('(c) Objective function')
set(gca,'position',[0.04 0.07 0.94 0.42])
grid on
hold off  
