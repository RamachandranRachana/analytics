function[]=logistic_regression(training_file,degree,test_file)
%% Training----------------------------------
A=load(training_file);
T=double(A(:,end));
deg=degree;
A=double(A(:,1:end-1));
%l=length(T);
T(T~=1)=0;
norows=size(A,1);
nocols=size(A,2);
x=zeros(norows,1);
x=x+1;

%deg 1
if deg==1
    phi=horzcat(x,A);
    weight=zeros(size(phi,2),1);
 %deg 2
else
    phi=horzcat(A(:,1),power(A(:,1),2));
    for i = 2:nocols
        phi=horzcat(phi,A(:,i));
        phi=horzcat(phi,power(A(:,i),2));
  
    end
    phi=horzcat(x,phi);
    weight=zeros(size(phi,2),1);
end
condition=true;
old_error=0;
n=0;
while condition
    y=zeros(size(phi,1),1);
    n=n+1;
    for j = 1:size(phi,1)   
        y(j, 1) = 1 / (1+exp(-(transpose(weight)*transpose(phi(j,1:end)))));   
    
    end
    %disp(y);
    R=zeros(size(y,1),size(y,1));
    for k=1:size(y,1)
        R(k,k)=y(k,1)*(1-y(k,1)) ;
    
    end    
    error_mat=transpose(phi)*(y-T);
    %disp(R);
    %disp(y);
    newweight=weight-(inv(transpose(phi)*R*phi)*error_mat);
    diffweight=sum(newweight)-sum(weight);
    differr=sum(error_mat)-old_error;
    %disp(abs(diffweight));
    %disp(abs(differr));
    if abs(diffweight)<0.001 && abs(differr)<0.001
        condition=false;
       
    end
    old_error=sum(error_mat);
    weight=newweight;
end
%disp(size(weight));
for d=1:size(weight,1)
   fprintf('w%d=%.4f\n',d-1,weight(d)); 
end
%% Testing---------------------------------------------
B=load(test_file);
TestT=double(B(:,end));
B=double(B(:,1:end-1));
TestT(TestT~=1)=0;
norowstest=size(B,1);
nocolstest=size(B,2);
xtest=zeros(norowstest,1);
xtest=xtest+1;
%deg 1
if deg==1
    phitest=horzcat(xtest,B);
    
 %deg 2
else
    phitest=horzcat(B(:,1),power(B(:,1),2));
    for i = 2:nocolstest
        phitest=horzcat(phitest,B(:,i));
        phitest=horzcat(phitest,power(B(:,i),2));
  
    end
    phitest=horzcat(xtest,phitest);
    
end

ytest=zeros(size(phitest,1),1);
pclass=zeros(size(phitest,1),1);
%disp(transpose(weight));
%disp(transpose(phitest));
tie=0;
for j = 1:size(phitest,1)
        wtphitest=transpose(weight)*transpose(phitest(j,1:end));
        %disp(wtphitest);
        yval = 1 / (1+exp(-(wtphitest)));
        %disp(yval);
        if(wtphitest>0 && yval>0.5)
           pclass(j,1)=1;
           ytest(j,1)=yval;
        elseif (wtphitest<0 && 1-yval>0.5)
            pclass(j,1)=0;
            ytest(j,1)=1-yval;
        else
            tie=1;
            pclass(j,1)=0;
            ytest(j,1)=1-yval;
        end
end
%disp(pclass);
caccuracy=0;
for j = 1:size(pclass,1)
   if pclass(j,1)==TestT(j,1)
       accuracy=1;
   elseif tie==1
       accuracy=0.5;
   else
       accuracy=0;
   end
   caccuracy=caccuracy+accuracy;
   fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',j,pclass(j,1) ,ytest(j,1),TestT(j,1), accuracy);
       
end
fprintf('classification accuracy=%6.4f\n', caccuracy/size(pclass,1));
end