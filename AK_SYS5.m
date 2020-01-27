%2019.12.22
%AK-Math 
% matlab 
% 1 call: all the values of local performance funciton 
% nps=1—parallel;   nps=2—series
% stopping condition for learning- Prob(C)>0.5; 
% select the local performance function to update 
clc
clear 
nmcs=50; 
for imcs=1:nmcs 

% clc
%clear
nn=10000;
nk=11;
nps=2;
nt=5; 
n1=12; 
n0=n1;
n3=1000; 
nfun=10; 
n1fun=zeros(nfun,1); 
ufun=zeros(nfun,1); 
sfun=zeros(nfun,1);
gs=zeros(nfun,1); 
xt=zeros(nt,nk,nfun);
mt=zeros(nfun,1);
mt(:)=nt;
gf=zeros(nt,nfun);
b=input1();
theta=zeros(nk,1);
lob=zeros(nk,1);
upb=zeros(nk,1);
for j=1:nk 
    theta(j)=1.0;
    lob(j)=1.0e-15;
    upb(j)=1.0e15; 
end 
%----------uniform distribution ------------
% x=rand(nn,nk);
% xt(:,:,1)=lhsdesign(nt,nk);
% for j=1:nk
%     for i=1:nn
%         x(i,j)=x(i,j)*(b(2,j)-b(1,j))+b(1,j);
%     end 
%     for i=1:nt 
%         xt(i,j,1)=xt(i,j,1)*b(j,2)+b(j,1); 
%     end 
% end 
%--------normal distribution ------------
x=randn(nn,nk); 
for j=1:nk 
    xt(:,j,1)=lhsnorm(0,1,nt);
end 
for j=1:nk 
    for i=1:nn 
        x(i,j)=x(i,j)*b(j,2)+b(j,1);
    end 
    for i=1:nt 
        xt(i,j,1)=xt(i,j,1)*b(j,2)+b(j,1); 
    end 
end
%----------------------------------------
for ik=2:nfun
    xt(:,:,ik)=xt(:,:,1);
end 
xs=zeros(n1,nk,nfun);
gg=zeros(n1,nfun); 
for i=1:n1 
    for ik=1:nfun
    	xs(i,:,ik)=x(i,:);
    end 
    gg(i,:)=response1(x(i,:)); 
end 

for i=1:nt 
    n0=n0+1;
    gf(i,:)=response1(xt(i,:,1));
end 

for ik=1:nfun 
    n1fun(ik)=n1; 
    ufun(ik)=1; 
end 
nw=0;
ssk=0;
gx=zeros(nn,nfun);
mse=zeros(nn,nfun);
gt=zeros(nt,1);
mset=zeros(nt,1);
ut=zeros(nn,1);
probc=zeros(nn,1);
var=zeros(nfun,1);

while 1
    nw=nw+1; 
    for ik=1:nfun 
        if ufun(ik)==1
            n1=n1fun(ik);
            xc=zeros(n1,nk);
            gc=zeros(n1,1); 
            for i=1:n1 
                for j=1:nk 
                    xc(i,j)=xs(i,j,ik);
                end 
                gc(i)=gg(i,ik); 
            end 
            [dmodel,perf]=dacefit(xc, gc, @regpoly0, @corrgauss, theta, lob, upb);
            if nw==1 || ik==ssk
                [gx(:,ik),mse(:,ik)] = predictor(x, dmodel);
                [gt,mset] = predictor(xt(1:mt(ik),:,ik), dmodel);
                var(ik)=sum((gf(1:mt(ik),ik)-gt(1:mt(ik))).^2);
            else 
                [gt,mset] = predictor(xt(1:mt(ik),:,ik), dmodel);
                c=sum((gf(1:mt(ik),ik)-gt(1:mt(ik))).^2);
                if c<var(ik)
                    [gx(:,ik),mse(:,ik)] = predictor(x, dmodel);
                    var(ik)=c; 
                else 
                    n1=n1fun(ik);
                    mt(ik)=mt(ik)+1;
                    gf(mt(ik),ik)=gg(n1,ik);
                    xt(mt(ik),:,ik)=xs(n1,:,ik);
                    gg(n1,ik)=0.0;
                    xs(n1,:,ik)=0.0; 
                    n1fun(ik)=n1fun(ik)-1;
                    
                end 
            end %nw==1 || ik==ssk
        end %ufun(ik)==1 
    end %ik
    
    for ik=1:nfun 
        ufun(ik)=0; 
        sfun(ik)=0; 
    end  
    mis=0;
    i1=0;
    j1=0;
    var2=1.0e8;
    gir=zeros(nn,1);
    iend=zeros(nn,1);
    for i=1:nn 
        %----------------identifiy the next local performance function ----
        k1=10000; 
        if nps==1 
            m=0;
            u=1.e10;
            u1=-1.0e10;
            for ik=1:nfun 
                if gx(i,ik)<0.0 && m==0 
                    c=abs(gx(i,ik))/sqrt(mse(i,ik));
                    if c<u 
                        u=c; 
                        k1=ik;
                    end
                else
                    m=1; 
                    if gx(i,ik)>0.0 
                        c=abs(gx(i,ik))/sqrt(mse(i,ik));
                        if c>u1 
                            u1=c; 
                            k1=ik; 
                            u=u1; 
                        end 
                    end 
                end 
            end % ik 
        else
            m=0; 
            u=1.0e10;
            u2=-1.0e10;
            for ik=1:nfun 
                if gx(i,ik)>0.0 && m==0 
                    c=abs(gx(i,ik))/sqrt(mse(i,ik));
                    if c<u 
                        u=c; 
                        k1=ik;
                    end 
                else
                    m=1;
                    if gx(i,ik)<0.0 
                        c=abs(gx(i,ik))/sqrt(mse(i,ik));
                        if c>u2
                            u2=c;
                            k1=ik;
                            u=u2; 
                        end 
                    end 
                end 
            end % ik 
        end % nps==1
        iend(i)=k1;
        %---------------------------------
        m=0;
        if k1~=10000 
            js=n1fun(k1);
            for j=1:js
                c=0.0;
                for n=1:nk 
                    c=c+(x(i,n)-xs(j,n,k1))^2;
                end 
                if c>0.0
                    c=sqrt(c)/nk;
                end 
                if c<1.0e-10
                    m=1;
                    break
                end 
            end 
        end 
        if u<var2 && m==0 
            var2=u;
            ss=i;
            ssk=k1; 
        end 
        %------------------compute the failure probability ----------------
        if nps==1 
            c=-1.0e10; 
            for ik=1:nfun 
                if gx(i,ik)>c 
                    k=ik;
                    c=gx(i,ik); 
                end 
            end 
        else 
            c=1.0e10;
            for ik=1:nfun 
                if gx(i,ik)<c 
                    k=ik;
                    c=gx(i,ik);
                end 
            end 
        end 
        gir(i)=k;
        if (gx(i,k)<0.0) 
            i1=i1+1;
        end 
        c=ResponseTrue1(x(i,:));
        if c<0.0 
            j1=j1+1; 
        end 
        c1=c*gx(i,k);
        if c1<0.0 
            mis=mis+1; 
        end 
    end %  i
    pf=i1/nn;
    pfs=j1/nn;
    %-------------------------------
    %new stopping condition 
    %-------------------------------
    uf=1.0; 
    var1=1.0e10;
    for i=1:nn 
        ik=iend(i);
        ut(i)=abs(gx(i,ik))/sqrt(mse(i,ik));
        if ~isinf(ut(i)) 
            c=normcdf(ut(i));
            probc(i)=c;
            uf=uf*c;
        end 
        m=mod(i,10000);
        if m==0 
            if var1>uf 
                var1=uf; 
            end 
            uf=1.0;
        end 
    end 
    c1=min(ut);
    c2=min(probc);
    display=[n0,pf,pfs,var2,var1,mis];
    disp(display);
    %------------------figure: the classification -------------------------
%     fx=zeros(i1,nk);
%     k=0;
%     for i=1:nn 
%         ik=gir(i);
%         if (gx(i,ik)<0.0) 
%             k=k+1; 
%             for j=1:nk 
%                 fx(k,j)=x(i,j);
%             end
%         end 
%     end 
%     s1=scatter(x(:,1),x(:,2),3,'c','filled');
%     hold on 
%     s2=scatter(fx(:,1),fx(:,2),3,'r','filled');
%     grid on 
%     axis([-4 4 -4 4]);
    if mis<100 
        i=1;
    end 
     if var1>=0.5 && n0>=20 
         if mis>100 
             i=1;
         end 
        break 
    end 
    close 
    %------------identify the status of each performance fucntion ---------
    for ik=1:nfun 
        if n1fun(ik)>=20
            ut=abs(gx(:,ik))./sqrt(mse(:,ik));
            var3=min(ut);
            if var3>=2.0 
                sfun(ik)=1;
            end 
        end
    end 
    %--------------filter the next performance function -----
    n0=n0+1;
    gs=response1(x(ss,:));
    for ik=1:nfun 
        js=0;
        if ik~=ssk 
            if sfun(ik)==1 
                continue 
            end 
            ike=n1fun(ik);
            for i=1:ike
                c=0.0;
                for j=1:nk 
                    c=c+(x(ss,j)-xs(i,j,ik))^2;
                end 
                if c>0.0 
                    c=sqrt(c)/nk;
                end 
                %****************************
                if c<1.0e-5
                %**********************************
                    js=1;
                    break 
                end 
            end 
        end 
        if js==0 
            i=1+n1fun(ik);
            for j=1:nk 
                xs(i,j,ik)=x(ss,j);
            end 
            gg(i,ik)=gs(ik);
            ufun(ik)=1; 
            n1fun(ik)=n1fun(ik)+1; 
        end 
    end % ik

    
    
    
end % while 
var1=(1.0-pf)/pf/nn;
var1=sqrt(var1)*1.0e2; 
display=[imcs,n0,pf,pfs,var1,mis];
disp(display)
fid=fopen('out5.dat','a');
fprintf(fid,'%f\t',display);
fprintf(fid,'\n');
fclose(fid);
fid=fopen('numfun5.dat','a');
fprintf(fid,'%f\t',imcs,n1fun);
fprintf(fid,'\n');
fclose(fid);
str1='---------------------------------------------';
disp(str1)
close 
end % imcs


i=1;






%--------------------------1111111111111111111-------------------------------
% 
% function b=input1()
%     b=zeros(2,2);
%     for j=1:2 
%         b(j,1)=0.0;
%         b(j,2)=1.0; 
%     end 
% end 
% 
% 
% function g=response1(x)
%         g(1)=8.0*x(2)^2-8.0*x(1)^2+(x(1)^2+x(2)^2)^2; 
%         g(2)=2.0*x(1)^2-2.0*x(2)^2-(x(1)^2+x(2)^2)^2; 
%         g(3)=8.0*x(2)^2-8.0*x(1)^2-(x(1)^2+x(2)^2)^2;
% end 
% 
% 
% 
% 
% function g=ResponseTrue1(x)
%     f(1)=8.0*x(2)^2-8.0*x(1)^2+(x(1)^2+x(2)^2)^2; 
%     f(2)=2.0*x(1)^2-2.0*x(2)^2-(x(1)^2+x(2)^2)^2; 
%     f(3)=8.0*x(2)^2-8.0*x(1)^2-(x(1)^2+x(2)^2)^2;
%     g=max(f);
% end 


% % ----------------------------22222222222222222222------------------------------

% function b=input1()
%     b(1,1)=2.0;
%     b(2,1)=5.0; 
%     for i=1:2 
%         b(i,2)=1.0; 
%     end 
% end 
% 
% 
% function g=response1(x)
%         g(1)=(x(1)^2+4.0)*(x(2)-1.0)/20.0-sin(5.0*x(1)/2.0)-2.0;
%         g(2)=(x(1)+2.0)^4-x(2)+4.0;
%         g(3)=(x(1)-4.0)^3-x(2)+2.0; 
% end 
% 
% 
% 
% 
% function g=ResponseTrue1(x)
%     f(1)=(x(1)^2+4.0)*(x(2)-1.0)/20.0-sin(5.0*x(1)/2.0)-2.0;
%     f(2)=(x(1)+2.0)^4-x(2)+4.0;
%     f(3)=(x(1)-4.0)^3-x(2)+2.0; 
%     g=max(f);
% end 





%----------------------------------333333333333333333333333333333333--------------------------------
% 
% function b=input1()
%     b=zeros(2,2);
%     for i=1:2 
%         b(i,2)=0.3; 
%     end 
% end 
% 
% 
% 
% 
% 
% function g=response1(x)
%         g(1)=(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+x(1)*x(2)+(-4.0+4.0*x(2)^2)*x(2)^2+0.8; 
%         g(2)=100.0*(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+100.0*x(1)*x(2)+100.0*(-4.0+4.0*x(2)^2)*x(2)^2+60.0; 
%         g(3)=500.0*(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+500.0*x(1)*x(2)+500.0*(-4.0+4.0*x(2)^2)*x(2)^2+250.0; 
% end 
% 
% 
% 
% 
% 
% function g=ResponseTrue1(x)
%     f(1)=(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+x(1)*x(2)+(-4.0+4.0*x(2)^2)*x(2)^2+0.8; 
%     f(2)=100.0*(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+100.0*x(1)*x(2)+100.0*(-4.0+4.0*x(2)^2)*x(2)^2+60.0; 
%     f(3)=500.0*(4.0-2.1*x(1)^2+x(1)^4/3.0)*x(1)^2+500.0*x(1)*x(2)+500.0*(-4.0+4.0*x(2)^2)*x(2)^2+250.0; 
%     g=max(f);
% end 

% 




%-----------------------------------444444444444444444444444------------------------------------

% 
% function b=input1()
%     b=zeros(2,2);
%     b(1,1)=-4;
%     b(2,1)=4;
%     b(1,2)=-20.0;
%     b(2,2)=2.0;
% end 
% 
% 
% function g=response1(x)
%         g(1)=sin(2.45*x(1))-(x(1)^2+3.9)*(x(2)+2.0)/20.0;
%         g(2)=100.0*sin(2.5*x(1))-5.0*(x(1)^2+4.0)*(x(2)+0.5);
%         g(3)=1000.0*sin(2.55*x(1))-5.0*(10.0*x(1)^2+41.0)*(x(2)-1);
% end 
% 
% 
% 
% 
% function g=ResponseTrue1(x)
%     f(1)=sin(2.45*x(1))-(x(1)^2+3.9)*(x(2)+2.0)/20.0;
%     f(2)=100.0*sin(2.5*x(1))-5.0*(x(1)^2+4.0)*(x(2)+0.5);
%     f(3)=1000.0*sin(2.55*x(1))-5.0*(10.0*x(1)^2+41.0)*(x(2)-1);
%     g=min(f);
% end 
% 






%----------------------------55555555555555555555555555---------------------------------


function b=input1()
    b=zeros(11,2);
    b(1,1)=0.5;
    b(2,1)=1.31;
    b(3,1)=0.5;
    b(4,1)=1.395;
    b(5,1)=0.875;
    b(6,1)=1.2;
    b(7,1)=0.4;
    b(8,1)=0.345;
    b(9,1)=0.192;
    b(1,2)=0.03;
    b(2,2)=0.03;
    b(3,2)=0.03;
    b(4,2)=0.03;
    b(5,2)=0.03;
    b(6,2)=0.03;
    b(7,2)=0.03;
    b(8,2)=0.006;
    b(9,2)=0.006;
    b(10,2)=10.0;
    b(11,2)=10.0;
    
end 



function g=response1(x)
        g(1)=1.0-(1.16-0.3717*x(2)*x(4)-0.00931*x(2)*x(10)-0.484*x(3)*x(9)+0.01343*x(6)*x(10)); 
        g(2)=4.01-(4.72-0.5*x(4)-0.19*x(2)*x(3)-0.0122*x(4)*x(10)+0.009325*x(6)*x(10)+0.000191*x(11)^2); 
        g(3)=32.0-(28.98+3.818*x(3)-4.2*x(1)*x(2)+0.0207*x(5)*x(10)+6.63*x(6)*x(9)-7.7*x(7)*x(8)+0.32*x(9)*x(10));
        g(4)=32.0-(33.86+2.95*x(3)+0.1792*x(10)-5.057*x(1)*x(2)-11.0*x(2)*x(8)-0.0215*x(5)*x(10)-9.98*x(7)*x(8)+22.0*x(8)*x(9));
        g(5)=32.0-(46.36-9.9*x(2)-12.9*x(1)*x(8)+0.1107*x(3)*x(10));
        g(6)=0.32-(0.261-0.0159*x(1)*x(2)-0.188*x(1)*x(8)-0.019*x(2)*x(7)+0.0144*x(3)*x(5)+0.0008757*x(5)*x(10));
        g(7)=0.32-(0.214+0.00817*x(5)-0.131*(1)*x(8)-0.0704*x(1)*x(9)+0.03099*x(2)*x(6)-0.018*x(2)*x(7)+0.0208*x(3)*x(8)+...
                        0.121*x(3)*x(9)-0.00364*x(5)*x(6)+0.0007715*x(5)*x(10)-0.0005354*x(6)*x(10)+0.00121*x(8)*x(11));
        g(8)=0.32-(0.74-0.61*x(2)-0.163*x(3)*x(8)+0.001232*x(3)*x(10)-0.166*x(7)*x(9)+0.227*x(2)^2);
        g(9)=9.9-(10.58-0.674*x(1)*x(2)-1.95*x(2)*x(8)+0.02054*x(3)*x(10)-0.0198*x(4)*x(10)+0.028*x(6)*x(10));
        g(10)=15.69-(16.45-0.489*x(3)*x(7)-0.843*x(5)*x(6)+0.0432*x(9)*x(10)-0.0556*x(9)*x(11)-0.000786*x(11)^2); 
    
end 




function g=ResponseTrue1(x)
    f(1)=1.0-(1.16-0.3717*x(2)*x(4)-0.00931*x(2)*x(10)-0.484*x(3)*x(9)+0.01343*x(6)*x(10)); 
    f(2)=4.01-(4.72-0.5*x(4)-0.19*x(2)*x(3)-0.0122*x(4)*x(10)+0.009325*x(6)*x(10)+0.000191*x(11)^2); 
    f(3)=32.0-(28.98+3.818*x(3)-4.2*x(1)*x(2)+0.0207*x(5)*x(10)+6.63*x(6)*x(9)-7.7*x(7)*x(8)+0.32*x(9)*x(10));
    f(4)=32.0-(33.86+2.95*x(3)+0.1792*x(10)-5.057*x(1)*x(2)-11.0*x(2)*x(8)-0.0215*x(5)*x(10)-9.98*x(7)*x(8)+22.0*x(8)*x(9));
    f(5)=32.0-(46.36-9.9*x(2)-12.9*x(1)*x(8)+0.1107*x(3)*x(10));
    f(6)=0.32-(0.261-0.0159*x(1)*x(2)-0.188*x(1)*x(8)-0.019*x(2)*x(7)+0.0144*x(3)*x(5)+0.0008757*x(5)*x(10));
    f(7)=0.32-(0.214+0.00817*x(5)-0.131*(1)*x(8)-0.0704*x(1)*x(9)+0.03099*x(2)*x(6)-0.018*x(2)*x(7)+0.0208*x(3)*x(8)+...
                        0.121*x(3)*x(9)-0.00364*x(5)*x(6)+0.0007715*x(5)*x(10)-0.0005354*x(6)*x(10)+0.00121*x(8)*x(11));
    f(8)=0.32-(0.74-0.61*x(2)-0.163*x(3)*x(8)+0.001232*x(3)*x(10)-0.166*x(7)*x(9)+0.227*x(2)^2);
    f(9)=9.9-(10.58-0.674*x(1)*x(2)-1.95*x(2)*x(8)+0.02054*x(3)*x(10)-0.0198*x(4)*x(10)+0.028*x(6)*x(10));
    f(10)=15.69-(16.45-0.489*x(3)*x(7)-0.843*x(5)*x(6)+0.0432*x(9)*x(10)-0.0556*x(9)*x(11)-0.000786*x(11)^2);
    g=min(f);
end 
















