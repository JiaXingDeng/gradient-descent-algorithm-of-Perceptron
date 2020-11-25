function[wt,t]=MP(x,y,r,w0)
%%
%ʹ������ݶ��½���SGD���ɵ����֪�������У�
% xָ����,yָ��Ӧ��ǩ��rָ�ݶ��½�ѧϰ�ʣ�w0Ϊ��ʼֵ������Ϊ[0,0,0]'��
% EXAMPLE��
% load('perceptron.mat');
%[wt,t]=MP(x,y,0.9,[1,1,1]')
Y=y;
X=[x',ones(15,1)];%���ݼ���
WX=X*w0;
bWX= WX > 0;
bY= Y == 1;
inx = bWX ~= Y';%�ж������Ƿ���ȷ
t=0;
error=[];
while sum(inx)>0 & t<10000
    tempx=X(inx,:);
    tempy=Y(inx);
    deltaw = tempx(1,:)'*tempy(1)';%ʹ�ô������ݵĵ�һ�����е���
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
    error(t) = -(X(inx,:)*w0/norm(w0))' * Y(inx)';%����ÿһ����loss����
end
%%
%��ͼ
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
